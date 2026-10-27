"""SAM3 text detection in frame chunks. Leaves ComfyUI core unchanged."""

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io


def _extract_text_prompts(conditioning, device, dtype):
    cond_meta = conditioning[0][1]
    multi = cond_meta.get("sam3_multi_cond")
    prompts = []
    if multi is not None:
        for entry in multi:
            emb = entry["cond"].to(device=device, dtype=dtype)
            mask = entry["attention_mask"]
            if mask is None:
                mask = torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
            else:
                mask = mask.to(device)
            prompts.append((emb, mask, entry.get("max_detections", 1)))
        return prompts

    emb = conditioning[0][0].to(device=device, dtype=dtype)
    mask = cond_meta.get("attention_mask")
    if mask is None:
        mask = torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
    else:
        mask = mask.to(device)
    return [(emb, mask, 1)]


def _coarse_mask(coarse, H, W):
    return (F.interpolate(coarse.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0] > 0).float()


def _prepare_refine(orig_image_hwc, coarse_mask, box_xyxy, H, W):
    pad_frac = 0.1
    x1, y1, x2, y2 = box_xyxy.tolist()
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad_frac))
    cy1 = max(0, int(y1 - bh * pad_frac))
    cx2 = min(W, int(x2 + bw * pad_frac))
    cy2 = min(H, int(y2 + bh * pad_frac))
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = orig_image_hwc[cy1:cy2, cx1:cx2, :3]
    crop_frame = comfy.utils.common_upscale(crop.unsqueeze(0).movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")

    mask_h, mask_w = coarse_mask.shape[-2:]
    mx1, my1 = int(cx1 / W * mask_w), int(cy1 / H * mask_h)
    mx2, my2 = int(cx2 / W * mask_w), int(cy2 / H * mask_h)
    if mx2 <= mx1 or my2 <= my1:
        return None
    mask_logit = coarse_mask[..., my1:my2, mx1:mx2].unsqueeze(0).unsqueeze(0)
    return crop_frame, mask_logit, cy1, cx1, cy2, cx2


def _scatter_refined(mask_logit, coarse_mask, cy1, cx1, cy2, cx2, H, W, device, dtype):
    refined_crop = F.interpolate(mask_logit, size=(cy2 - cy1, cx2 - cx1), mode="bilinear", align_corners=False)
    full_mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    full_mask[:, :, cy1:cy2, cx1:cx2] = refined_crop
    coarse_full = F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
    return ((full_mask[0] > 0) | (coarse_full[0] > 0)).float()


def _refine_masks_batch(sam3_model, image, jobs, H, W, device, dtype, iterations, refine_batch):
    out = [None] * len(jobs)
    if iterations <= 0:
        for i, (_, coarse, _) in enumerate(jobs):
            out[i] = _coarse_mask(coarse, H, W)
        return out

    pending = []
    for i, (frame_index, coarse, box) in enumerate(jobs):
        prepared = _prepare_refine(image[frame_index], coarse, box, H, W)
        if prepared is None:
            out[i] = _coarse_mask(coarse, H, W)
            continue
        pending.append((i, coarse, prepared))

    step = max(1, refine_batch)
    for start in range(0, len(pending), step):
        group = pending[start:start + step]
        crops = torch.cat([item[2][0] for item in group], dim=0).to(device=device, dtype=dtype)
        logits = [
            F.interpolate(item[2][1].to(device=device, dtype=dtype), size=(1008, 1008), mode="bilinear", align_corners=False)
            for item in group
        ]
        mask_logit = torch.cat(logits, dim=0)
        for _ in range(iterations):
            coarse_input = F.interpolate(mask_logit, size=(1008, 1008), mode="bilinear", align_corners=False)
            mask_logit = sam3_model.forward_segment(crops, mask_inputs=coarse_input)
        for j, (i, coarse, prepared) in enumerate(group):
            _, _, cy1, cx1, cy2, cx2 = prepared
            out[i] = _scatter_refined(mask_logit[j:j + 1], coarse, cy1, cx1, cy2, cx2, H, W, device, dtype)
    return out


class SAM3_DetectBatch(io.ComfyNode):
    """Text-prompt SAM3 detection that runs several frames in one forward."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_DetectBatch",
            display_name="SAM3 Detect Batch",
            category="image/detection",
            search_aliases=["sam3", "batch", "video", "face"],
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Conditioning.Input("conditioning", display_name="conditioning", optional=True, tooltip="Text conditioning from CLIPTextEncode"),
                io.Float.Input("threshold", display_name="threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Int.Input("refine_iterations", display_name="refine_iterations", default=1, min=0, max=5, tooltip="SAM decoder refinement passes. 0 uses the raw detector mask."),
                io.Boolean.Input("individual_masks", display_name="individual_masks", default=False, tooltip="Output per-object masks instead of a union"),
                io.Int.Input("frame_batch", display_name="frame_batch", default=8, min=1, max=32, tooltip="Frames per forward. This does not reduce the amount of work; lower it if VRAM runs out."),
            ],
            outputs=[
                io.Mask.Output("masks"),
                io.BoundingBox.Output("bboxes"),
            ],
        )

    @classmethod
    def execute(cls, model, image, conditioning=None, threshold=0.5, refine_iterations=1, individual_masks=False, frame_batch=8) -> io.NodeOutput:
        B, H, W, _ = image.shape
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model
        cond_list = _extract_text_prompts(conditioning, device, dtype) if conditioning is not None and len(conditioning) > 0 else []
        chunk = max(1, int(frame_batch))
        idev = comfy.model_management.intermediate_device()
        all_bbox_dicts = []
        all_masks = []
        pbar = comfy.utils.ProgressBar(B)

        for start in range(0, B, chunk):
            end = min(B, start + chunk)
            frames = comfy.utils.common_upscale(
                image[start:end, ..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled"
            ).to(device=device, dtype=dtype)
            chunk_boxes = [[] for _ in range(end - start)]
            chunk_masks = [[] for _ in range(end - start)]
            refine_jobs = []
            refine_slots = []

            # One ViT pass for every text prompt in this chunk. face and head used to
            # each run the backbone, and a bigger frame_batch does not reduce that work.
            detector = sam3_model.detector
            trunk_out = detector.backbone["vision_backbone"].trunk(frames)
            resizer = detector.backbone["language_backbone"]["resizer"]
            del frames
            for text_embeddings, text_mask, max_det in cond_list:
                prompt_mask = text_mask.bool() if text_mask is not None else None
                results = detector.forward_from_trunk(trunk_out, resizer(text_embeddings), prompt_mask)
                box_scale = results["boxes"].new_tensor([W, H, W, H])
                for local in range(end - start):
                    pred_boxes = results["boxes"][local] * box_scale
                    scores = results["scores"][local]
                    masks = results["masks"][local]
                    probs = scores.sigmoid()
                    keep = probs > threshold
                    kept_boxes = pred_boxes[keep].cpu()
                    kept_scores = probs[keep].cpu()
                    kept_masks = masks[keep]
                    order = kept_scores.argsort(descending=True)[:max_det]
                    kept_boxes = kept_boxes[order]
                    kept_scores = kept_scores[order]
                    kept_masks = kept_masks[order]
                    for box, score in zip(kept_boxes, kept_scores):
                        chunk_boxes[local].append({
                            "x": float(box[0]), "y": float(box[1]),
                            "width": float(box[2] - box[0]), "height": float(box[3] - box[1]),
                            "score": float(score),
                        })
                    if kept_masks.shape[0] == 0:
                        continue
                    if individual_masks:
                        for coarse, box in zip(kept_masks, kept_boxes):
                            refine_slots.append(local)
                            refine_jobs.append((start + local, coarse, box))
                    else:
                        union_box = torch.stack([
                            kept_boxes[:, 0].min(), kept_boxes[:, 1].min(),
                            kept_boxes[:, 2].max(), kept_boxes[:, 3].max(),
                        ])
                        refine_slots.append(local)
                        refine_jobs.append((start + local, kept_masks.max(dim=0).values, union_box))
            del trunk_out

            if refine_jobs:
                refined = _refine_masks_batch(
                    sam3_model, image, refine_jobs, H, W, device, dtype, refine_iterations, chunk)
                for local, mask in zip(refine_slots, refined):
                    chunk_masks[local].append(mask)

            for local in range(end - start):
                all_bbox_dicts.append(chunk_boxes[local])
                frame_masks = chunk_masks[local]
                if frame_masks:
                    combined = torch.cat(frame_masks, dim=0)
                    if individual_masks:
                        all_masks.append(combined)
                    else:
                        all_masks.append((combined > 0).any(dim=0).float())
                elif individual_masks:
                    all_masks.append(torch.zeros(0, H, W, device=idev))
                else:
                    all_masks.append(torch.zeros(H, W, device=idev))
            pbar.update(end - start)

        all_masks = [mask.to(idev) for mask in all_masks]
        mask_out = torch.cat(all_masks, dim=0) if individual_masks else torch.stack(all_masks)
        return io.NodeOutput(mask_out, all_bbox_dicts)


class SAM3FrameBatchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SAM3_DetectBatch]


async def comfy_entrypoint() -> SAM3FrameBatchExtension:
    return SAM3FrameBatchExtension()
