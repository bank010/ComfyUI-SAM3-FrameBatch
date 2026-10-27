# ComfyUI-SAM3-FrameBatch

把 SAM 3.1 的文字检测改成一次前向处理多帧。每一帧仍然单独出掩码，切镜不会沿用上一镜的结果。

节点名：`SAM3 Detect Batch`（`SAM3_DetectBatch`）。

`frame_batch` 默认 8。显存不够时改成 4 或 2。提示词用法和原来的 SAM3 Detect 一样，例如 `face:8, head:6`。

把本目录放到 `ComfyUI/custom_nodes/ComfyUI-SAM3-FrameBatch`，重启 ComfyUI。
