# ComfyUI-SAM3-FrameBatch

把 SAM3 / SAM3.1 的文字检测改成一次前向处理多帧。每一帧仍然单独出掩码，切镜不会沿用上一镜的结果。

节点名：`SAM3 Detect Batch`（`SAM3_DetectBatch`），分类 `image/detection`。

## 安装

把本目录放到 `ComfyUI/custom_nodes/ComfyUI-SAM3-FrameBatch`，重启 ComfyUI。

需要包含 `SAM3_Detect` / `SAM3 Video Track` 的 ComfyUI。本仓库在 ComfyUI v0.43.0（2026-10-26）的源码上核对过 `forward_from_trunk`、`forward_segment` 和 SAM3.1 的 tracker 接口。

## 用法

`CLIPTextEncode`（SAM3 CLIP）接到 `conditioning`。提示词语法和核心的 `SAM3 Detect` 一致：

```text
face:8, head:6
```

`face:8` 表示 face 这一类最多保留 8 个检测，`:N` 只限制检测数量，不改变阈值。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `threshold` | 0.5 | 检测分数阈值 |
| `refine_iterations` | 1 | SAM decoder 细化次数；0 表示直接用检测器掩码 |
| `individual_masks` | false | 每个目标一张掩码，而不是每帧一张并集 |
| `frame_batch` | 8 | 每次前向的帧数；显存不够时改成 4 或 2 |

输出：

- `masks`：`individual_masks=false` 时是 `[帧数, H, W]`；`true` 时按目标展平，帧与目标的对应关系与核心节点一致，不保证 `masks.shape[0] == 帧数`。
- `bboxes`：`[每帧的 [检测字典]]`，字典包含 `x`、`y`、`width`、`height`、`score`。

## 这次优化了什么

### 1. 细化时不再重复跑 ViT

原来 `refine_iterations=2` 会调用两次 `forward_segment()`，而 `forward_segment()` 内部每次都重新跑整条视觉主干。实际上图像没变，只有 mask prompt 在变，编码结果可以复用：

```text
旧：crop 编码 → decoder → crop 编码 → decoder
新：crop 编码 → decoder → decoder
```

代码里的 `_encode_segment_features()` 就是 `SAM3Model.forward_segment()` 去掉 mask decoder 的那一半，逐行对应（SAM3 的 `scalp` 裁剪、SAM3.1 的 `interactive` 分支、`no_mem` embedding 都保留）。如果以后 ComfyUI 改了内部接口，会自动退回旧的 `forward_segment()` 路径。

### 2. 重叠的提示词共用一张 crop

`face:8, head:6` 这种写法，原来每个提示词、每帧各跑一次 crop 编码，哪怕两个框几乎完全重叠。现在同一帧里包含关系超过 80% 的框会合并成一个 crop，细化完再取并集；位置相距很远的框仍然各用各的 crop，不会为了省时间把整幅画面塞进一个 1008×1008。

### 3. 显存和内存

- 检测结果里的整张 `[B, 200, h, w]` 掩码在细化前就 `del` 掉（8 帧、fp16 时约 265 MB）。
- 全分辨率掩码不再跨 chunk 留在显存：旧版要等整个视频跑完才搬到 CPU（1000 帧 1080p 约 8.3 GB 显存），现在每个 chunk 直接写入 CPU 缓冲。
- crop 保持原始尺寸，批处理时再上采样，不再为每个待细化目标常驻一张 1008×1008 的 float32 crop。
- 细化结果按批次搬回 CPU，不再把整块视频的全分辨率掩码留在显存里。
- 并集输出直接写进预分配的 `[B, H, W]` 缓冲，不再先存列表再 `torch.stack`，长视频的主机内存峰值从约 2 份输出降到 1 份。

### 4. 健壮性

- 空 batch 直接返回 `[0, H, W]`，不再在 `torch.stack` 处崩溃。
- 没接 `conditioning` 时明确报错，而不是静默输出全零掩码（这个节点没有 box/point 输入，没提示词就没有可做的事）。
- 非 SAM3 模型给出明确错误。
- 每个 chunk、每批细化都会检查 ComfyUI 的取消信号，长视频可以中断。
- 文本 embedding 的 `resizer` 只跑一次，不再每个 chunk 重算。

## 先测再调

在启动 ComfyUI 前设置 `SAM3_FRAMEBATCH_TIMING=1`：

```bash
# Linux / macOS
SAM3_FRAMEBATCH_TIMING=1 python main.py
# Windows
set SAM3_FRAMEBATCH_TIMING=1 && python main.py
```

跑一次后会打印一行：

```text
[SAM3 Detect Batch] frames=100 chunk=8 prompts=2 iters=1 individual=False dtype=torch.float16 total=... load=... pre=... trunk=... detect=... refine=... post=...
```

怎么看：

- `trunk` 是检测阶段的 ViT，通常是最大头。
- `refine` 是细化阶段的 crop 编码 + decoder。它接近 `total` 的一半时，把 `refine_iterations` 改成 0 就能直接减半。
- `load` 很大说明模型在反复加载/offload，不是算子慢。
- `dtype=torch.float32` 说明没跑半精度，速度会差很多。
- `pre` 很大说明瓶颈在 CPU 预处理，优先降输入分辨率。

## 减少耗时的顺序

1. **`refine_iterations=0`**：省掉细化阶段的整次 ViT，通常直接减半。代价是掩码来自检测器的 288×288，边缘更粗；先对比一版，能接受就用。
2. **确认 `individual_masks=false`**：true 会按目标逐个出 crop，目标一多就成倍变慢。
3. **确认精度和 offload**：看上面的 `dtype`；再检查启动日志有没有 `lowvram` / `loaded partially`。显存够时用 `--highvram`，避免每次调用重新加载权重。
4. **减少提示词数量**：每个提示词都要单独跑一遍检测头和一次细化 crop。`face:8, head:6` 如果只需要脸的掩码，就只留 `face`。
5. **`frame_batch` 实测 4 / 8 / 16**：它主要影响显存，不改变总计算量；批量太小会浪费 GPU，太大则 OOM。显存修好后可以试着往上调。
6. **输入先降到 1080p 或 720p**：ViT 固定 1008，降分辨率不会减少 ViT 计算，但能减少 CPU 预处理、crop 内存和最终掩码的写出时间。
7. **抽帧**：如果 24fps 不需要逐帧掩码，每 2 帧检测一次再插值，耗时直接减半；切镜处要单独处理。
8. **启动参数**：不要强制 `--use-split-cross-attention`；NVIDIA 上可以试 `--fast fp16_accumulation autotune`（可能小幅提速，属实验项）。`--force-fp16` 只在 `dtype` 不是 fp16 时用。

SAM3 的视觉主干在 1008×1008 下每帧约 5 TFLOPs，100 帧就是几百 TFLOPs。4090 级别通常十几到二十几秒，3060 级别一分多钟是正常量级；如果你的机器远高于这个量级，优先查 `dtype`、offload 和启动参数，而不是继续调 `frame_batch`。

## 性能预期

下表是每个 chunk（8 帧）里 1008×1008 ViT 前向的次数，不含检测头、decoder 和预处理。每次前向都是 batch=8：

| 场景 | 旧 | 新 |
| --- | --- | --- |
| 1 个提示词，1 次细化 | 2 | 2 |
| 1 个提示词，2 次细化 | 3 | 2 |
| 2 个重叠提示词，1 次细化 | 3 | 2 |
| 2 个重叠提示词，2 次细化 | 5 | 2 |
| 2 个分离提示词，2 次细化 | 5 | 3 |

单提示词、单次细化时速度基本不变，收益主要在显存和内存；多提示词、多次细化时细化部分的编码开销按上表下降。实际耗时还取决于 GPU、分辨率和 `individual_masks`，建议用自己的素材对比。

调参建议：

- `frame_batch` 控制峰值显存。检测和细化各有一批 batch=8 的 1008 输入，OOM 时先降到 4 或 2。
- 现在 `refine_iterations=2` 只多一次 decoder，不再多一次 ViT，边缘要求高时值得试。
- `individual_masks=true` 会按目标逐个出 crop，目标多时明显更慢；只在需要逐目标掩码时打开。

## 已知取舍

- 并集输出仍然沿用“每个提示词一个 union crop”的做法。`face:8` 遇到满屏人脸时，8 张脸会共用一个覆盖全屏的 crop，边界精度不如核心节点逐检测细化。需要更高质量时用 `individual_masks=true`，或者把提示词拆成更小的区域分别跑。
- 掩码细化只在 `refine_iterations>0` 时进行；0 是检测器的 288×288 掩码直接上采样，最快但边缘最粗。
- 内部接口（`detector.backbone`、`forward_from_trunk`）不是 ComfyUI 的稳定 API。节点已经做了 fallback，但 ComfyUI 大改 SAM3 时仍可能需要跟进。

## 测试

测试不加载权重、不需要 ComfyUI，用假的 SAM3 模型检查形状、阈值、空输入、crop 合并、批大小，以及“编码次数”是否符合预期：

```bash
python -m unittest discover -s tests -v
```

18 个用例在 CPU 上约 0.2 秒跑完。测试环境需要 PyTorch，可以直接用 ComfyUI 自带的 Python。

更详细的代码审查和逐条问题记录见 [REVIEW.md](REVIEW.md)。
