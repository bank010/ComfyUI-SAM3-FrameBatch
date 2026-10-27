# 代码审查与优化记录

审查对象：`bank010/ComfyUI-SAM3-FrameBatch`，commit `095e419`（2026-10-27）。
对照实现：ComfyUI `426f5ee`（v0.43.0）的 `comfy_extras/nodes_sam3.py`、`comfy/ldm/sam3/detector.py`、`comfy/ldm/sam3/sam.py`。

## 结论

整体方向是对的：把 B 帧拼成一个 batch 过一次 ViT，是这类视频检测节点最大的性能来源；逐帧独立解码也保住了切镜不串结果的需求。

真正的浪费在**细化阶段**，不在检测阶段：

1. `refine_iterations=2` 时，同一张 crop 会被完整编码两次（`forward_segment()` 每次都重跑视觉主干）。
2. 多个提示词（`face:8, head:6`）时，同一帧会按提示词各跑一次 crop 编码，即使两个框几乎重叠。
3. 整个 `[B, 200, h, w]` 的检测掩码在细化期间一直活着；全分辨率结果也一直留在显存里。
4. 长视频的并集输出先存 list 再 `torch.stack`，主机内存峰值是输出的两倍。

前两项按 ViT 前向次数计算，典型场景可以从 5 次降到 2 次；后两项是显存/内存问题。

## 问题清单

### P0：健壮性

| # | 位置（旧版） | 问题 | 处理 |
| --- | --- | --- | --- |
| 1 | `execute` L131–L220 | `B == 0` 时 `torch.stack([])` 直接抛异常 | 早退，返回 `[0, H, W]` |
| 2 | L136 | `conditioning` 为空时静默输出全零掩码，用户很难发现接错了 | 明确抛 `ValueError` |
| 3 | L135 | 非 SAM3 模型在 `sam3_model.detector` 处抛 `AttributeError` | 提前检查并给出可读错误 |
| 4 | L143、L87–L100 | 整个 chunk/细化过程没有取消检查，长视频只能杀进程 | 每个 chunk、每批细化调用 `throw_exception_if_processing_interrupted()` |

### P1：性能与显存

| # | 位置（旧版） | 问题 | 处理 |
| --- | --- | --- | --- |
| 5 | L95–L97 | 每次细化都调用 `forward_segment()`，内部重跑整条 ViT；图像没变，特征可以复用 | 编码一次，循环里只跑 SAM decoder |
| 6 | L159–L194 | 每个提示词每帧各建一个 crop，重叠的 `face`/`head` 会重复编码同一块区域 | 同帧内按包含关系合并 crop，细化后取并集 |
| 7 | L159–L195 | `results`（`[B, 200, 288, 288]`，8 帧 fp16 约 250 MB）在细化期间仍被引用 | 提取完 box/mask 后 `del results`，并清掉循环里最后一个 `masks` 视图 |
| 8 | L52 | 每个待细化 crop 立刻上采样到 1008×1008 float32（12 MB/个）并常驻 | crop 保持原尺寸（超过 1008 时先缩到 1008），按细化批次准备 |
| 9 | L96 | `mask_logit` 已经是 1008×1008，循环里又 `F.interpolate` 到 1008，纯空转 | 删除 |
| 10 | L218–L219 | 全分辨率掩码跨 chunk 一直留在显存，直到整个视频结束才 `.to(idev)`；并集输出再 `torch.stack` 复制一份 | 每个 chunk 直接写进预分配 CPU 缓冲，显存只保留当前 chunk |
| 11 | L157/L161 | `resizer(text_embeddings)` 每个 chunk 重算 | 循环外只算一次 |

关于第 8 项的数量级：一个 200×200 的人脸 crop 原尺寸约 0.48 MB，放大到 1008² float32 是 12.2 MB；`frame_batch=8`、两个提示词时，旧版在细化开始前就要常驻约 195 MB 的预处理 crop。新版每批只准备 8 张，超大 crop 先缩到 1008²，最坏情况约 97 MB；人脸这种小 crop 每批只有几 MB。

关于第 10 项的数量级：1000 帧 1080p 的 float32 掩码约 8.3 GB。旧版在整段视频结束前把这些掩码全部留在显存（1000 帧 1080p 约 8.3 GB 显存），再搬到 CPU 并用 `torch.stack` 复制一份（CPU 峰值约 16.6 GB）。新版每帧写入预分配 CPU 缓冲，显存只保留当前 chunk，CPU 峰值约 8.3 GB（输出本身无法避免）。`individual_masks=true` 因为目标数不定，仍在 CPU 累积后用 `torch.cat` 拼接，但不再占显存。

### P2：正确性/质量上的取舍

| # | 位置 | 说明 |
| --- | --- | --- |
| 12 | L188–L194 | 并集模式下，每个提示词只做**一个** union crop，而不是像核心节点那样逐检测细化。`face:8` 在满屏人脸时 8 张脸共用一个覆盖全屏的 crop，单脸有效分辨率下降，边缘会比核心节点粗。这是原项目的性能取舍，本次没有改变单提示词行为，只把跨提示词的重复 crop 合并掉。需要逐目标质量时用 `individual_masks=true`。 |
| 13 | L184–L187 | `individual_masks=true` 时按目标出 crop，目标多时细化成本线性增长；输出行数也不再等于帧数。与核心节点行为一致，但文档需要写清楚。 |
| 14 | L12–L33 | `_extract_text_prompts` 是从核心节点复制的；ComfyUI 改 `sam3_multi_cond` 结构时会静默漂移。已在 README 里注明依赖内部接口。 |
| 15 | L156/L161 | 依赖 `detector.backbone[...]` 和 `forward_from_trunk`，不是稳定 API。已加 `forward_segment` fallback：`_encode_segment_features()` 取不到内部结构时自动退回旧路径。 |

### 核对过、不是问题的点

- `results["boxes"]` 是**归一化 xyxy**（`forward_from_trunk` 文档和 `box_cxcywh_to_xyxy` 的实现都确认了），乘 `[W, H, W, H]` 再算 `width = x2 - x1` 是对的。
- 检测器 `forward(..., threshold=...)` 里的 `threshold` 参数在实现中并未使用，节点自己按 `sigmoid(score) > threshold` 过滤，与核心节点一致。
- `max_det` 的默认值 1、`kept_scores.argsort(descending=True)[:max_det]` 的顺序、按提示词分组的行为，都与核心 `SAM3_Detect` 一致。
- 掩码用 logits `> 0` 而不是用户阈值，这是 SAM 的惯例，核心节点也如此。
- `del results` 是安全的：`kept_masks = masks[keep]` 是高级索引，已经拷贝；后续 `kept_masks[order]` 也是拷贝。但循环变量 `masks` 是视图，需要一起 `del`，否则 `[B, 200, 288, 288]` 的存储不会释放。

## 改动摘要

`__init__.py`：

- 新增 `_encode_segment_features()` / `_segment_from_features()`：把 `SAM3Model.forward_segment()` 拆成“编码一次 + decoder 多次”。逐行对应核心实现，包括：
  - SAM3 非 multiplex 的 `need_tracker=True` 和 `scalp` 裁剪；
  - SAM3.1 multiplex 的 `tracker_mode="interactive"`；
  - `interactivity_no_mem_embed` / `no_mem_embed` 的加法；
  - 只传 `mask_inputs`，`multimask_output=False`。
- 新增 `_group_crops()`：同帧跨提示词按“交集 / 较小框面积 ≥ 0.8”合并 crop，远景目标不合并。
- `_prepare_refine()` 不再提前放大 crop。
- `_refine_masks_batch()` 每批只编码一次；结果直接 `.cpu()`；去掉空转的 interpolate。
- `execute()`：预分配并集输出、按帧写入、空 batch 早退、缺 conditioning/非 SAM3 报错、取消检查、`del results`/`del masks`、`resizer` 提前。
- 输出形状、bbox 字典格式、`individual_masks` 语义保持兼容。

`tests/test_frame_batch.py`：

- 17 个 CPU 用例，不加载权重；
- 假模型可以验证“编码调用次数”，从而直接锁住第 5、6、11 项优化；
- fast path 与 fallback 输出逐元素相等；
- 覆盖 SAM3 非 multiplex 和 SAM3.1 multiplex 两条编码分支。

## 没有做的、可以继续做的

1. **OOM 自动降级**：捕获 `torch.cuda.OutOfMemoryError` 后把当前 chunk 对半重试。实现不难，但会改变进度条语义，且错误可能来自别处；先保持“显存不够就调 `frame_batch`”的可预测行为。
2. **提示词内部的 union crop 拆分**：`face:8` 目前还是一个大 crop。可以按框的连通性做空间聚类，给相距远的检测各一个 crop；质量更接近核心节点，代价是目标多时变慢。建议做成输入开关，而不是默认行为。
3. **大框跳过细化**：框覆盖画面比例很高时，crop 和整帧几乎一样，细化收益有限。可以按面积阈值跳过，但会改变掩码，需实测。
4. **跨提示词复用 FPN/位置编码**：`forward_from_trunk()` 每个提示词都重算 `convs` 和 `position_encoding`。改调私有 `detector._detect()` 可以省 3%–5%，但耦合更深，收益不大。
5. **`frame_batch` 与 `refine_batch` 分开**：目前细化 batch 等于 `frame_batch`。检测和细化对显存的敏感度不同，分开可以更细地调，但会多一个参数。
6. **示例工作流 + 基准脚本**：用同一段 100 帧 1080p 素材，固定 seed，分别跑旧/新、`refine_iterations=1/2`、单/双提示词，用 `torch.cuda.Event` 记录 `execute()` 时间。当前环境没有 GPU 和 SAM3 权重，无法给出真实数字，README 里只给了 ViT 前向次数的理论上限。
7. **工程化**：`LICENSE`（pyproject 写的是 GPL-3.0-or-later，但仓库没有 LICENSE 文件）、GitHub Actions 跑 CPU mock 测试、英文 README、版本号从 1.0.0 升到 1.1.0。

## 验证

```text
$ python -m unittest discover -s tests -v
Ran 17 tests in 0.212s
OK
```

没有在真实 SAM3 权重和 GPU 上跑过；本环境没有 ComfyUI 运行时和模型文件。因此：

- 数值等价性靠“逐行对照 `forward_segment` + fallback 测试”保证；
- 性能收益靠“ViT 前向次数”推导，不声称具体秒数；
- 建议合并前在目标 GPU 上用真实素材跑一次 A/B，重点看 `refine_iterations=2` 和 `face:8, head:6` 两个场景。
