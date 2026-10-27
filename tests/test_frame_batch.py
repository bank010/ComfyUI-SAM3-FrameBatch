"""Unit tests for SAM3_DetectBatch using a fake SAM3 model.

These tests run without ComfyUI or model weights:

    python -m unittest discover -s tests -v

They check the node contract (shapes, boxes, empty inputs) and, more
importantly, how many times the expensive vision encoder is called. The fake
tracker produces the same mask whether the encoder output is reused or
recomputed, so the fast path and the fallback path can be compared directly.
"""

import importlib.util
import sys
import types
import unittest
import weakref
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Minimal ComfyUI stubs
# ---------------------------------------------------------------------------

class _ComfyNode:
    pass


class _ComfyExtension:
    pass


class _SpecType:
    @staticmethod
    def Input(*args, **kwargs):
        return None

    @staticmethod
    def Output(*args, **kwargs):
        return None


class _Schema:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _NodeOutput:
    def __init__(self, *args):
        self.args = args


def _install_comfy_stubs():
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_model_gpu = lambda model: None
    mm.get_torch_device = lambda: torch.device("cpu")
    mm.intermediate_device = lambda: torch.device("cpu")
    mm.throw_exception_if_processing_interrupted = lambda: None

    utils = types.ModuleType("comfy.utils")
    utils.common_upscale = lambda samples, width, height, method, crop: F.interpolate(
        samples, size=(height, width), mode=method
    )

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.value = 0

        def update(self, n):
            self.value += n

    utils.ProgressBar = ProgressBar

    comfy.model_management = mm
    comfy.utils = utils

    api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyNode = _ComfyNode
    latest.ComfyExtension = _ComfyExtension
    latest.io = types.SimpleNamespace(
        ComfyNode=_ComfyNode,
        Schema=_Schema,
        NodeOutput=_NodeOutput,
        Model=_SpecType,
        Image=_SpecType,
        Conditioning=_SpecType,
        Float=_SpecType,
        Int=_SpecType,
        Boolean=_SpecType,
        Mask=_SpecType,
        BoundingBox=_SpecType,
    )
    api.latest = latest

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.utils"] = utils
    sys.modules["comfy_api"] = api
    sys.modules["comfy_api.latest"] = latest


def _load_node_module():
    _install_comfy_stubs()
    spec = importlib.util.spec_from_file_location("sam3_frame_batch", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NODE = _load_node_module()


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

class FakeVisionBackbone:
    multiplex = False

    def __init__(self):
        self.trunk_calls = 0
        self.encode_calls = 0
        self.encoded_batches = []
        self.levels = 3

    def trunk(self, frames):
        self.trunk_calls += 1
        return torch.zeros(frames.shape[0], 4, 2, 2)

    def __call__(self, images, need_tracker=False, tracker_mode=None):
        self.encode_calls += 1
        self.encoded_batches.append(images.shape[0])
        b = images.shape[0]
        features = [torch.ones(b, 4, 4, 4) for _ in range(self.levels)]
        return None, None, features, None


class FakeTracker:
    def __init__(self):
        self.head_calls = 0
        self.head_batches = []
        self.interactivity_no_mem_embed = torch.zeros(1, 1, 4)

    def _forward_sam_heads(self, backbone_features, mask_inputs=None, high_res_features=None,
                           multimask_output=False):
        self.head_calls += 1
        self.head_batches.append(mask_inputs.shape[0])
        b = mask_inputs.shape[0]
        value = mask_inputs.mean(dim=(1, 2, 3), keepdim=True)
        # [B,1,1,1] -> [B,1,1008,1008], same value for every pixel.
        return None, value.expand(b, 1, 1008, 1008).clone(), None, None


class FakeDetector:
    """Returns fixed detections per prompt.

    ``specs`` maps the scalar stored in a prompt embedding to a list of
    ``(box_xyxy_pixels, score_logit)`` pairs.
    """

    def __init__(self, H, W, specs):
        self.H = H
        self.W = W
        self.scalp = 0
        self.specs = specs
        self.vision = FakeVisionBackbone()
        self.backbone = {
            "vision_backbone": self.vision,
            "language_backbone": {"resizer": torch.nn.Identity()},
        }

    def forward_from_trunk(self, trunk_out, text_embeddings, text_mask):
        b = trunk_out.shape[0]
        key = round(float(text_embeddings.flatten()[0]), 3)
        spec = self.specs[key]
        q = len(spec)
        boxes = torch.zeros(b, q, 4)
        scores = torch.zeros(b, q)
        mask_h, mask_w = 24, 36  # detector masks are much smaller than 1008
        masks = torch.zeros(b, q, mask_h, mask_w)
        for i, (box, score) in enumerate(spec):
            x1, y1, x2, y2 = box
            boxes[:, i] = torch.tensor([
                x1 / self.W, y1 / self.H, x2 / self.W, y2 / self.H,
            ])
            scores[:, i] = score
            mx1, my1 = int(x1 / self.W * mask_w), int(y1 / self.H * mask_h)
            mx2, my2 = int(x2 / self.W * mask_w), int(y2 / self.H * mask_h)
            masks[:, i, my1:max(my2, my1 + 1), mx1:max(mx2, mx1 + 1)] = 2.0
        return {"boxes": boxes, "scores": scores, "masks": masks}


class FakeSAM3Model:
    def __init__(self, H, W, specs):
        self.detector = FakeDetector(H, W, specs)
        self.vision = self.detector.vision
        self.tracker = FakeTracker()
        self.forward_segment_calls = 0
        self.segment_batches = []

    def forward_segment(self, images, mask_inputs=None):
        self.forward_segment_calls += 1
        self.segment_batches.append(mask_inputs.shape[0])
        b = mask_inputs.shape[0]
        value = mask_inputs.mean(dim=(1, 2, 3), keepdim=True)
        return value.expand(b, 1, 1008, 1008).clone()


class _InnerModel:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model

    def get_dtype(self):
        return torch.float32


class FakeModelPatcher:
    def __init__(self, diffusion_model):
        self.model = _InnerModel(diffusion_model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 64, 96


def make_image(batch):
    return torch.rand(batch, H, W, 3)


def make_conditioning(keys, max_dets=None):
    multi = []
    for i, key in enumerate(keys):
        multi.append({
            "cond": torch.full((1, 3, 4), float(key)),
            "attention_mask": None,
            "max_detections": max_dets[i] if max_dets else 1,
        })
    return [[multi[0]["cond"].clone(), {"sam3_multi_cond": multi}]]


def default_specs():
    return {
        1.0: [((10, 10, 30, 30), 3.0)],
        2.0: [((12, 8, 34, 36), 3.0)],   # overlaps 1.0
        3.0: [((5, 5, 20, 20), 3.0), ((60, 30, 90, 55), 2.0)],  # two objects
        4.0: [((60, 30, 90, 55), 3.0)],  # far from 1.0
        5.0: [((10, 10, 30, 30), -3.0)],  # below threshold
    }


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.sam = FakeSAM3Model(H, W, default_specs())
        self.model = FakeModelPatcher(self.sam)

    def run_node(self, batch=1, keys=(1.0,), max_dets=None, **kwargs):
        params = dict(threshold=0.5, refine_iterations=1, individual_masks=False, frame_batch=8)
        params.update(kwargs)
        return NODE.SAM3_DetectBatch.execute(
            self.model, make_image(batch), make_conditioning(keys, max_dets), **params
        )

    # -- contract -----------------------------------------------------------

    def test_single_prompt_shapes(self):
        masks, boxes = self.run_node().args
        self.assertEqual(tuple(masks.shape), (1, H, W))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(boxes[0]), 1)
        self.assertAlmostEqual(boxes[0][0]["x"], 10.0, places=4)
        self.assertAlmostEqual(boxes[0][0]["width"], 20.0, places=4)
        self.assertGreater(masks.sum().item(), 0)

    def test_multiple_frames(self):
        masks, boxes = self.run_node(batch=5).args
        self.assertEqual(tuple(masks.shape), (5, H, W))
        self.assertEqual(len(boxes), 5)

    def test_threshold_drops_detection(self):
        masks, boxes = self.run_node(keys=(5.0,)).args
        self.assertEqual(len(boxes[0]), 0)
        self.assertEqual(masks.sum().item(), 0)

    def test_empty_batch(self):
        masks, boxes = self.run_node(batch=0).args
        self.assertEqual(tuple(masks.shape), (0, H, W))
        self.assertEqual(boxes, [])

    def test_missing_conditioning_raises(self):
        with self.assertRaises(ValueError):
            NODE.SAM3_DetectBatch.execute(self.model, make_image(1), None)

    def test_refine_zero_skips_decoder(self):
        masks, _ = self.run_node(refine_iterations=0).args
        self.assertEqual(self.sam.vision.encode_calls, 0)
        self.assertEqual(self.sam.tracker.head_calls, 0)
        self.assertEqual(self.sam.forward_segment_calls, 0)
        self.assertGreater(masks.sum().item(), 0)

    def test_detector_masks_freed_before_refine(self):
        """The [B, Q, h, w] mask tensor must not stay alive during refinement."""
        refs = []
        original_forward = self.sam.detector.forward_from_trunk

        def forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            refs.append(weakref.ref(result["masks"]))
            return result

        self.sam.detector.forward_from_trunk = forward

        original_refine = NODE._refine_masks_batch
        freed_at_refine = []

        def refine(*args, **kwargs):
            freed_at_refine.append(refs[0]() is None)
            return original_refine(*args, **kwargs)

        NODE._refine_masks_batch = refine
        try:
            self.run_node()
        finally:
            NODE._refine_masks_batch = original_refine
            self.sam.detector.forward_from_trunk = original_forward
        self.assertEqual(freed_at_refine, [True])

    # -- encoder reuse ------------------------------------------------------

    def test_iterations_reuse_one_encode(self):
        self.run_node(refine_iterations=3)
        self.assertEqual(self.sam.vision.encode_calls, 1)
        self.assertEqual(self.sam.tracker.head_calls, 3)
        self.assertEqual(self.sam.forward_segment_calls, 0)

    def test_multiplex_backbone_uses_interactive_features(self):
        self.sam.vision.multiplex = True
        masks, _ = self.run_node(refine_iterations=2).args
        self.assertEqual(tuple(masks.shape), (1, H, W))
        self.assertEqual(self.sam.vision.encode_calls, 1)
        self.assertEqual(self.sam.tracker.head_calls, 2)

    def test_sam3_scalp_is_sliced(self):
        self.sam.vision.levels = 4
        self.sam.detector.scalp = 1
        masks, _ = self.run_node(refine_iterations=1).args
        self.assertEqual(tuple(masks.shape), (1, H, W))
        self.assertEqual(self.sam.vision.encode_calls, 1)
        self.assertEqual(self.sam.tracker.head_calls, 1)

    def test_fallback_matches_fast_path(self):
        fast, _ = self.run_node(refine_iterations=2).args

        original = NODE._encode_segment_features
        NODE._encode_segment_features = lambda *args, **kwargs: None
        try:
            self.sam.vision.encode_calls = 0
            self.sam.tracker.head_calls = 0
            slow, _ = self.run_node(refine_iterations=2).args
        finally:
            NODE._encode_segment_features = original

        self.assertEqual(self.sam.vision.encode_calls, 0)
        self.assertEqual(self.sam.forward_segment_calls, 2)
        self.assertTrue(torch.equal(fast, slow))

    # -- crop grouping ------------------------------------------------------

    def test_overlapping_prompts_share_a_crop(self):
        masks, boxes = self.run_node(keys=(1.0, 2.0)).args
        self.assertEqual(len(boxes[0]), 2)
        self.assertEqual(self.sam.vision.encode_calls, 1)
        self.assertEqual(self.sam.vision.encoded_batches, [1])
        self.assertEqual(self.sam.tracker.head_calls, 1)

    def test_distant_prompts_keep_separate_crops(self):
        self.run_node(keys=(1.0, 4.0))
        self.assertEqual(self.sam.vision.encode_calls, 1)
        self.assertEqual(self.sam.vision.encoded_batches, [2])

    def test_refine_batches_are_limited(self):
        # 2 frames x 2 distant prompts = 4 jobs, refine_batch = frame_batch = 2.
        self.run_node(batch=2, keys=(1.0, 4.0), frame_batch=2)
        self.assertEqual(self.sam.vision.encode_calls, 2)
        self.assertEqual(self.sam.vision.encoded_batches, [2, 2])

    def test_group_crops_unit(self):
        a = torch.zeros(1, 1, 10, 10)
        box_a = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box_inside = torch.tensor([2.0, 2.0, 8.0, 8.0])
        box_far = torch.tensor([50.0, 50.0, 60.0, 60.0])

        merged = NODE._group_crops([(a, box_a), (a, box_inside)])
        self.assertEqual(len(merged), 1)

        separate = NODE._group_crops([(a, box_a), (a, box_far)])
        self.assertEqual(len(separate), 2)

    # -- individual masks ---------------------------------------------------

    def test_individual_masks(self):
        masks, boxes = self.run_node(keys=(3.0,), max_dets=[2], individual_masks=True).args
        self.assertEqual(len(boxes[0]), 2)
        self.assertEqual(masks.shape[0], 2)
        self.assertEqual(tuple(masks.shape[1:]), (H, W))
        self.assertEqual(self.sam.vision.encode_calls, 1)
        self.assertEqual(self.sam.vision.encoded_batches, [2])

    def test_no_detections_individual(self):
        masks, boxes = self.run_node(keys=(5.0,), individual_masks=True).args
        self.assertEqual(tuple(masks.shape), (0, H, W))
        self.assertEqual(len(boxes[0]), 0)


if __name__ == "__main__":
    unittest.main()
