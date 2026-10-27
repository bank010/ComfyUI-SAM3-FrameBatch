"""SAM3 text detection in frame chunks. Leaves ComfyUI core unchanged."""

import os
import time

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io

# SAM3's ViT uses 72x72 patches with RoPE, so the encoder input is always 1008.
ENCODER_SIZE = 1008
# Same crop padding as ComfyUI's SAM3_Detect node.
CROP_PAD = 0.1
# Two crops are merged when the smaller box is at least this much inside the
# larger one. "face" and "head" share a crop; distant objects keep their own.
MERGE_CONTAINMENT = 0.8
# Opt-in profiling: SAM3_FRAMEBATCH_TIMING=1 prints a per-stage breakdown.
_TIMING = os.environ.get("SAM3_FRAMEBATCH_TIMING") == "1"


def _clock(device):
    """Wall clock that also waits for async GPU work, for honest stage timings."""
    if not _TIMING:
        return 0.0
    if device is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
    return time.perf_counter()


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


def _frames_to_detect(image, max_skip, change_threshold):
    """Detect on a frame, then reuse that mask while the picture stays similar.

    A cut or other large change is detected immediately. max_skip also forces a
    new detection so a slow head turn cannot keep an old mask for long.
    """
    count = image.shape[0]
    if max_skip <= 0 or count <= 1:
        return list(range(count))
    small = F.interpolate(image[..., :3].movedim(-1, 1), size=(64, 64), mode="area")
    diffs = (small[1:] - small[:-1]).abs().flatten(1).mean(1)
    chosen = [0]
    since = 0
    for index, diff in enumerate(diffs.tolist(), start=1):
        since += 1
        if since > max_skip or diff >= change_threshold:
            chosen.append(index)
            since = 0
    return chosen


def _coarse_mask(coarse, H, W):
    return (F.interpolate(coarse.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0] > 0).float()


def _box_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _containment(a, b):
    """Intersection over the smaller box. 1.0 means one box contains the other."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    smaller = min(_box_area(a), _box_area(b))
    return (iw * ih) / smaller if smaller > 0 else 0.0


def _union_box(a, b):
    return torch.cat((torch.minimum(a[:2], b[:2]), torch.maximum(a[2:], b[2:])))


def _group_crops(parts):
    """Merge overlapping crops for one frame.

    ``parts`` is a list of ``(coarse_mask, box_xyxy)`` from each text prompt.
    The union output only needs one mask per frame, so prompts looking at the
    same object should share one refinement crop instead of running the crop
    encoder once per prompt. Boxes that do not overlap stay separate so a
    distant object still gets its own high-resolution crop.
    """
    if len(parts) <= 1:
        return list(parts)
    groups = []
    for coarse, box in sorted(parts, key=lambda p: -_box_area(p[1])):
        for group in groups:
            if _containment(group[1], box) >= MERGE_CONTAINMENT:
                group[0] = torch.maximum(group[0], coarse)
                group[1] = _union_box(group[1], box)
                break
        else:
            groups.append([coarse, box])
    return [(coarse, box) for coarse, box in groups]


def _encode_segment_features(sam3_model, images):
    """Run the vision encoder once for a batch of 1008x1008 crops.

    This mirrors ``SAM3Model.forward_segment`` up to the SAM mask decoder.
    The encoder output only depends on the image, so iterative refinement can
    reuse it instead of running the whole ViT once per iteration. Returns
    ``None`` when the installed ComfyUI does not expose the expected internals;
    callers then fall back to ``forward_segment`` per iteration.
    """
    detector = getattr(sam3_model, "detector", None)
    tracker = getattr(sam3_model, "tracker", None)
    if detector is None or tracker is None or not hasattr(tracker, "_forward_sam_heads"):
        return None
    try:
        bb = detector.backbone["vision_backbone"]
    except (AttributeError, KeyError, TypeError):
        return None
    if not hasattr(bb, "multiplex"):
        return None

    try:
        with torch.no_grad():
            if bb.multiplex:
                _, _, tracker_features, _ = bb(images, tracker_mode="interactive")
            else:
                _, _, tracker_features, _ = bb(images, need_tracker=True)
                scalp = getattr(detector, "scalp", 0)
                if scalp > 0:
                    tracker_features = tracker_features[:-scalp]
            if not tracker_features:
                return None
            high_res = list(tracker_features[:-1])
            backbone_feat = tracker_features[-1]
            no_mem = getattr(tracker, "interactivity_no_mem_embed", None)
            if no_mem is None:
                no_mem = getattr(tracker, "no_mem_embed", None)
            if no_mem is not None:
                B, C, H, W = backbone_feat.shape
                feat_flat = backbone_feat.flatten(2).permute(0, 2, 1)
                feat_flat = feat_flat + no_mem.to(device=feat_flat.device, dtype=feat_flat.dtype)
                backbone_feat = feat_flat.view(B, H, W, C).permute(0, 3, 1, 2)
            return backbone_feat, high_res
    except (AttributeError, KeyError, TypeError):
        return None


def _segment_from_features(sam3_model, features, mask_inputs):
    backbone_feat, high_res = features
    _, high_res_masks, _, _ = sam3_model.tracker._forward_sam_heads(
        backbone_features=backbone_feat,
        mask_inputs=mask_inputs,
        high_res_features=high_res,
        multimask_output=False,
    )
    return high_res_masks


def _prepare_refine(orig_image_hwc, coarse_mask, box_xyxy, H, W):
    pad_frac = CROP_PAD
    x1, y1, x2, y2 = box_xyxy.tolist()
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad_frac))
    cy1 = max(0, int(y1 - bh * pad_frac))
    cx2 = min(W, int(x2 + bw * pad_frac))
    cy2 = min(H, int(y2 + bh * pad_frac))
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    # Keep the crop at its native size and upscale while batching. A 1008x1008
    # float32 crop is 12 MB; preparing every job up front adds up quickly.
    crop = orig_image_hwc[cy1:cy2, cx1:cx2, :3].movedim(-1, 0).contiguous()
    if max(crop.shape[-2:]) > ENCODER_SIZE:
        # Anything larger is downscaled at batch time anyway; doing it here
        # bounds the memory of the pending crops.
        crop = F.interpolate(crop.unsqueeze(0), size=(ENCODER_SIZE, ENCODER_SIZE),
                             mode="bilinear", align_corners=False)[0]

    mask_h, mask_w = coarse_mask.shape[-2:]
    mx1, my1 = int(cx1 / W * mask_w), int(cy1 / H * mask_h)
    mx2, my2 = int(cx2 / W * mask_w), int(cy2 / H * mask_h)
    if mx2 <= mx1 or my2 <= my1:
        return None
    mask_logit = coarse_mask[..., my1:my2, mx1:mx2].unsqueeze(0).unsqueeze(0)
    return crop, mask_logit, cy1, cx1, cy2, cx2


def _scatter_refined(mask_logit, coarse_mask, cy1, cx1, cy2, cx2, H, W, device, dtype):
    refined_crop = F.interpolate(mask_logit, size=(cy2 - cy1, cx2 - cx1), mode="bilinear", align_corners=False)
    full_mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    full_mask[:, :, cy1:cy2, cx1:cx2] = refined_crop
    coarse_full = F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
    return ((full_mask[0] > 0) | (coarse_full[0] > 0)).float()


def _refine_masks_batch(sam3_model, image, jobs, H, W, device, dtype, iterations, refine_batch):
    """Refine coarse detector masks. Returns a list of CPU float masks."""
    out = [None] * len(jobs)
    if not jobs:
        return out
    if iterations <= 0:
        for i, (_, coarse, _) in enumerate(jobs):
            out[i] = _coarse_mask(coarse, H, W).cpu()
        return out

    step = max(1, int(refine_batch))
    for start in range(0, len(jobs), step):
        comfy.model_management.throw_exception_if_processing_interrupted()
        group = []
        for i in range(start, min(start + step, len(jobs))):
            frame_index, coarse, box = jobs[i]
            prepared = _prepare_refine(image[frame_index], coarse, box, H, W)
            if prepared is None:
                out[i] = _coarse_mask(coarse, H, W).cpu()
            else:
                group.append((i, coarse, prepared))
        if not group:
            continue

        crops = torch.cat([
            F.interpolate(item[2][0].unsqueeze(0), size=(ENCODER_SIZE, ENCODER_SIZE),
                          mode="bilinear", align_corners=False)
            for item in group
        ], dim=0).to(device=device, dtype=dtype)

        # The per-crop coarse masks have different sizes, so upscale before cat.
        mask_logit = torch.cat([
            F.interpolate(item[2][1].to(device=device, dtype=dtype), size=(ENCODER_SIZE, ENCODER_SIZE),
                          mode="bilinear", align_corners=False)
            for item in group
        ], dim=0)

        # Encode the crops once, then run only the cheap SAM decoder per pass.
        features = _encode_segment_features(sam3_model, crops)
        with torch.no_grad():
            for _ in range(iterations):
                if features is not None:
                    mask_logit = _segment_from_features(sam3_model, features, mask_logit)
                else:
                    mask_logit = sam3_model.forward_segment(crops, mask_inputs=mask_logit)
        del features

        for j, (i, coarse, prepared) in enumerate(group):
            _, _, cy1, cx1, cy2, cx2 = prepared
            out[i] = _scatter_refined(
                mask_logit[j:j + 1], coarse, cy1, cx1, cy2, cx2, H, W, device, dtype
            ).cpu()
        del crops, mask_logit
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
                io.Int.Input("frame_batch", display_name="frame_batch", default=8, min=1, max=32, tooltip="Frames packed into one forward. Lower it if VRAM runs out."),
                io.Int.Input("max_skip", display_name="max_skip", default=0, min=0, max=30, tooltip="0 detects every frame. Above 0 reuses the last mask on similar frames and can miss motion in a different video."),
                io.Float.Input("change_threshold", display_name="change_threshold", default=0.06, min=0.0, max=1.0, step=0.01, tooltip="Frame difference that forces a new detection. Lower it if motion is being missed."),
            ],
            outputs=[
                io.Mask.Output("masks"),
                io.BoundingBox.Output("bboxes"),
            ],
        )

    @classmethod
    def execute(cls, model, image, conditioning=None, threshold=0.5, refine_iterations=1, individual_masks=False, frame_batch=8, max_skip=0, change_threshold=0.06) -> io.NodeOutput:
        chosen = _frames_to_detect(image, max_skip, change_threshold)
        full_count = image.shape[0]
        if len(chosen) < full_count:
            image = image[chosen]
        B, H, W, _ = image.shape
        idev = comfy.model_management.intermediate_device()

        if B == 0:
            return io.NodeOutput(torch.zeros(0, H, W, device=idev), [])
        if conditioning is None or len(conditioning) == 0:
            raise ValueError("SAM3 Detect Batch needs text conditioning (CLIPTextEncode with a SAM3 CLIP).")

        load_start = time.perf_counter() if _TIMING else 0.0
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        timing = None
        if _TIMING:
            now = _clock(device)
            timing = {"load": now - load_start, "pre": 0.0, "trunk": 0.0,
                      "detect": 0.0, "refine": 0.0, "post": 0.0}
            mark = now
        sam3_model = getattr(model.model, "diffusion_model", None)
        if sam3_model is None or not hasattr(sam3_model, "detector"):
            raise ValueError("SAM3 Detect Batch needs a SAM3/SAM3.1 checkpoint.")

        cond_list = _extract_text_prompts(conditioning, device, dtype)
        resizer = sam3_model.detector.backbone["language_backbone"]["resizer"]
        # The prompt embeddings are the same for every chunk; resize them once.
        prompts = [(resizer(emb), mask, max_det) for emb, mask, max_det in cond_list]

        chunk = max(1, int(frame_batch))
        # ``image`` only contains the frames selected by _frames_to_detect, but
        # the outputs always cover the full input batch. Skipped frames repeat
        # the last detected mask/boxes.
        all_bbox_dicts = [None] * full_count
        all_masks = [] if individual_masks else None
        if not individual_masks:
            # Writing into one buffer avoids keeping a per-frame list and then
            # copying it all again in torch.stack (2x peak RAM on long videos).
            mask_out = torch.empty((full_count, H, W), dtype=torch.float32, device=idev)
        pbar = comfy.utils.ProgressBar(B)

        for start in range(0, B, chunk):
            comfy.model_management.throw_exception_if_processing_interrupted()
            end = min(B, start + chunk)
            n = end - start
            frames = comfy.utils.common_upscale(
                image[start:end, ..., :3].movedim(-1, 1), ENCODER_SIZE, ENCODER_SIZE, "bilinear", crop="disabled"
            ).to(device=device, dtype=dtype)
            if timing is not None:
                now = _clock(device)
                timing["pre"] += now - mark
                mark = now

            chunk_boxes = [[] for _ in range(n)]
            union_parts = [[] for _ in range(n)]
            individual_jobs = []
            individual_slots = []

            # One ViT pass for every text prompt in this chunk. face and head used to
            # each run the backbone, and a bigger frame_batch does not reduce that work.
            detector = sam3_model.detector
            trunk_out = detector.backbone["vision_backbone"].trunk(frames)
            del frames
            if timing is not None:
                now = _clock(device)
                timing["trunk"] += now - mark
                mark = now
            for text_embeddings, text_mask, max_det in prompts:
                prompt_mask = text_mask.bool() if text_mask is not None else None
                results = detector.forward_from_trunk(trunk_out, text_embeddings, prompt_mask)
                box_scale = results["boxes"].new_tensor([W, H, W, H])
                for local in range(n):
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
                            individual_slots.append(local)
                            individual_jobs.append((start + local, coarse, box))
                    else:
                        union_box = torch.stack([
                            kept_boxes[:, 0].min(), kept_boxes[:, 1].min(),
                            kept_boxes[:, 2].max(), kept_boxes[:, 3].max(),
                        ])
                        union_parts[local].append((kept_masks.max(dim=0).values, union_box))
                # The full [B, num_queries, h, w] mask tensor can be hundreds of
                # MB; do not keep it alive while the refinement runs. The last
                # ``masks`` view from the loop would keep that storage alive, so
                # drop it together with the result dict.
                del masks
                del results
            del trunk_out
            if timing is not None:
                now = _clock(device)
                timing["detect"] += now - mark
                mark = now

            if individual_masks:
                jobs, slots = individual_jobs, individual_slots
            else:
                jobs, slots = [], []
                for local in range(n):
                    for coarse, box in _group_crops(union_parts[local]):
                        slots.append(local)
                        jobs.append((start + local, coarse, box))

            frame_masks = [[] for _ in range(n)]
            if jobs:
                refined = _refine_masks_batch(
                    sam3_model, image, jobs, H, W, device, dtype, refine_iterations, chunk)
                for local, mask in zip(slots, refined):
                    frame_masks[local].append(mask)
            if timing is not None:
                now = _clock(device)
                timing["refine"] += now - mark
                mark = now

            for local in range(n):
                original_index = chosen[start + local]
                all_bbox_dicts[original_index] = chunk_boxes[local]
                masks = frame_masks[local]
                if individual_masks:
                    if masks:
                        all_masks.append(torch.cat(masks, dim=0))
                    else:
                        all_masks.append(torch.zeros(0, H, W, device=idev))
                elif masks:
                    combined = torch.cat(masks, dim=0)
                    mask_out[original_index] = (combined > 0).any(dim=0).float()
                else:
                    mask_out[original_index].zero_()
            pbar.update(n)
            if timing is not None:
                now = _clock(device)
                timing["post"] += now - mark
                mark = now

        if individual_masks:
            expanded_masks = [None] * full_count
            cursor = 0
            last_mask = None
            for index in range(full_count):
                if cursor < B and chosen[cursor] == index:
                    last_mask = all_masks[cursor]
                    cursor += 1
                expanded_masks[index] = last_mask
            mask_out = torch.cat(expanded_masks, dim=0) if expanded_masks else torch.zeros(0, H, W, device=idev)
        else:
            cursor = 0
            last_mask = None
            for index in range(full_count):
                if cursor < B and chosen[cursor] == index:
                    last_mask = mask_out[index]
                    cursor += 1
                else:
                    mask_out[index].copy_(last_mask)

        last_boxes = None
        for index in range(full_count):
            if all_bbox_dicts[index] is not None:
                last_boxes = all_bbox_dicts[index]
            else:
                all_bbox_dicts[index] = last_boxes

        if timing is not None:
            now = _clock(device)
            timing["post"] += now - mark
            print(
                f"[SAM3 Detect Batch] frames={full_count} detect_frames={B} chunk={chunk} "
                f"prompts={len(prompts)} iters={refine_iterations} individual={individual_masks} "
                f"skip={max_skip} dtype={dtype} total={now - load_start:.2f}s "
                f"load={timing['load']:.2f}s pre={timing['pre']:.2f}s "
                f"trunk={timing['trunk']:.2f}s detect={timing['detect']:.2f}s "
                f"refine={timing['refine']:.2f}s post={timing['post']:.2f}s",
                flush=True,
            )
        return io.NodeOutput(mask_out, all_bbox_dicts)


class SAM3FrameBatchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SAM3_DetectBatch]


async def comfy_entrypoint() -> SAM3FrameBatchExtension:
    return SAM3FrameBatchExtension()
