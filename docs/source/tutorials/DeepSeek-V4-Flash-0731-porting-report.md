# DeepSeek-V4-Flash-0731 移植修改与阶段成果

## 1. 目标、基线与结论

目标是在百度昆仑 vLLM-Kunlun `0.25.1` 基础上，参考 upstream vLLM `v0.25.1`
已经具备的 DeepSeek-V4-Flash-0731 模型拓扑和调度接口，使官方模型能在单机 8 卡
P800 上加载和提供 OpenAI API。实现没有采用仓库 PR #402 的代码。

截至 2026-08-16，成果应准确表述为：

- 已完成模型注册、权重转换、P800 reference attention/mHC、MoE routing、平台约束
  和 vLLM 0.25.1 运行时兼容层；
- 已完成一次全量官方 checkpoint 转换，并在 8 张 P800 上完成完整加载；
- `/health` 和 `/v1/models` 正常，冷启动后的首个贪心请求可得到正确答案；
- 连续第二个独立请求仍会出现 token id 0 / 重复 BOS，logprobs 路径会出现非有限值；
- 所以当前是“启动、加载、首请求链路打通”，不是“生产部署完成”。

## 2. 提交脉络

主要代码提交如下，后面的文档和自动化脚本是对这些工作的固化：

| 提交 | 作用 |
|---|---|
| `7ab5f42` | 首版模型、attention、mHC、converter、MoE 与平台支持 |
| `6e91784` | 运行时兼容、schema 和 converter 对旧 PyTorch float8 的适配 |
| `9d53e59` | 实机加载后发现的 attention、block table、KV bind 等修复 |
| `845ad0e` | 修复 dense block-FP8 的 UE8M0 scale 解码 |
| `294971b` | 按 vLLM 0.25.1 的真实构造入口修正 KVBlockZeroer 补丁 |

PR #402 没有被当成实现来源或正确性依据。模型结构以 upstream vLLM 0.25.1 的
DeepSeek V4 实现为语义基线，再针对 Kunlun OOT platform 和 P800 可用算子做适配。

## 3. 代码修改明细

### 3.1 模型注册与 upstream 适配

- `vllm_kunlun/models/__init__.py`：注册 DeepSeek V4 architecture，让模型配置能
  解析到 Kunlun adapter。
- `vllm_kunlun/models/deepseek_v4.py`：复用 upstream 的模型拓扑和参数映射，替换
  XPU 上不能直接工作的执行缝隙；保留 DeepSeek hash expert routing 所需信息；
  将本版本范围限制在主模型，未宣称 DSpark/MTP 已可用。
- `vllm_kunlun/schema.py`：接受 DeepSeek V4 新增的配置字段，避免插件 schema 在
  进入模型构建之前错误拒绝官方 config。

这里采用 adapter/monkey patch 而不是复制整个 upstream 模型文件，目的是让参数名、
权重映射和 vLLM 0.25.1 调度语义尽量仍由 upstream 代码负责。

### 3.2 P800 attention、压缩器与 mHC

- `vllm_kunlun/models/deepseek_v4_attention.py`：加入 P800 eager/reference 路径，
  覆盖 RoPE/RMSNorm、compressed slot 和 metadata 构建、sparse indexer、BF16 sparse
  attention、compressor，以及连续/带 padding cache 的访问。upstream 依赖但当前
  `kunlun_ops` 不提供或签名不兼容的 fused kernel 会落到 PyTorch 实现。
- `vllm_kunlun/models/deepseek_v4_mhc.py`：加入 torch-native mHC，实现当前设备上
  可执行的 residual mixing 路径。

这些 reference 实现首先解决功能连通性，不代表最终性能形态。Graph、长上下文和
所有 fused kernel 的性能优化都不在当前已验证范围内。

### 3.3 W8A8 与 MoE

- `vllm_kunlun/quantization/compressed_tensors/compressed_tensors_moe.py`：补齐
  `sqrtsoftplus` routing、基于 input IDs 的 hash expert routing 和 shared expert
  W8A8 路径，使模型的 MoE 语义不被现有通用路由逻辑错误替代。
- `tools/convert_deepseek_v4_0731_w8a8.py`：按 safetensors shard 流式转换官方权重，
  支持 MXFP4 expert 和 F8_E4M3 dense 权重，产出 compressed-tensors 的动态
  per-token / per-channel INT8 模型，并写入 `kunlun_w8a8_manifest.json`。

转换器中最关键的实机修复是 `F8_E8M0`：P800 环境所用 PyTorch 2.5 不能把这种
scale dtype 当数值 float 正确载入，因此代码直接读取 safetensors payload 的原始
byte，再按 UE8M0 定义计算 `2^(byte-127)`。旧逻辑把原始字节 115–121 当普通数值，
曾产生最大约 419.78 的错误 weight scale，最终导致 NaN/token 0。修复后完整模型的
最大 scale 是约 0.1348425。

### 3.4 vLLM 0.25.1 / Kunlun 运行时兼容

- `vllm_kunlun/__init__.py`：集中安装 torch 2.5、Triton 和部分 warmup/monitor 的
  兼容补丁，保证它们在 worker fork/import 顺序下生效。
- `vllm_kunlun/platforms/kunlun.py`：为本模型设置并校验 TP8、BF16 KV、block 256、
  32K、eager 等边界，对未实现组合尽早给出明确错误。
- `vllm_kunlun/v1/worker/block_table.py`：当现有 `kunlun_ops` 没有
  `compute_slot_mappings` 时使用 NumPy fallback，避免 engine 初始化失败。
- `vllm_kunlun/v1/worker/utils.py`：允许同层索引绑定多个 DeepSeek cache entry，
  并让 KVBlockZeroer 覆盖 vLLM 0.25.1 实际调用的 `__init__` API。zeroer 的 eager
  清理仅在 upstream cache config 请求 zeroing 时工作；“强制给所有 MLA cache
  zero”已做过 A/B，不能修复当前连续请求问题，因此没有把该实验开关作为方案。

### 3.5 测试与文档

- `tests/ut/test_deepseek_v4_converter.py`：覆盖 MXFP4、block-FP8、UE8M0 raw byte、
  tensor companion/recipe 等转换行为；本地 8 项测试通过。
- `tools/deepseek_v4_flash_0731/`：新增基础环境、并发下载、转换、checkpoint 校验、
  启停、连续请求验收和诊断采集脚本。
- 本报告、空白机部署指南和当前机器接手文档分别记录“改了什么”“如何复现”以及
  “下一位工程师从哪里继续”。

## 4. 实机成果与证据

### 4.1 权重转换

完整转换 manifest：

```json
{
  "source": "/root/donlin_model/DeepSeek-V4-Flash-0731",
  "quantized": 35721,
  "copied": 878,
  "mxfp4": 35328,
  "block_fp8": 390,
  "output_bytes": 306124999420
}
```

输出共 48 个 shard，目录显示约 286 GiB。扫描全部 weight scale 得到：最小
`1.7301305e-05`，最大 `0.1348425`，zero=0，nonfinite=0。这个扫描是判断 converter
是否用了正确 UE8M0 语义的必要门槛。

### 4.2 8 卡加载与服务

在 8 × P800 上实测：

- 完整模型加载约 17.6 秒，engine 初始化约 8.4 秒；
- 每张卡模型占用约 36.13 GiB；
- 不同诊断配置下每卡总显存约 86.97–90.14 GiB；
- KV cache 约 39.27 GiB，可容纳约 67045 tokens；
- 32K 上下文估算并发约 2.05；
- `/health` 返回 200；`/v1/models` 显示模型名
  `deepseek-v4-flash-0731`、max length 32768。

### 4.3 生成验证矩阵

| 检查 | 结果 | 解读 |
|---|---|---|
| 完整模型加载 | 通过 | architecture、权重映射和显存分配已连通 |
| `/health`、`/v1/models` | 通过 | API 和 engine 就绪，不代表数值正确 |
| 冷启动后首请求 | 可通过 | 曾正确回答“法国的首都是巴黎”“中国的首都是北京” |
| 同进程第二个独立请求 | 失败 | token id 0、重复 `<｜begin▁of▁sentence｜>` |
| `logprobs` | 失败 | 非有限 float 无法编码为 JSON，返回 400 |
| `--no-async-scheduling` A/B | 无修复 | 排除 async scheduling 是唯一原因 |
| 强制 KV zero A/B | 无修复 | 8 worker 均注册 62 tensors，问题仍复现 |

因此，`/health=200`、显存占用正常或“首请求答对”都不能单独作为验收结果。

## 5. 当前限制与下一步

最高优先级是定位“首请求之后产生或遗留的非有限状态”。建议按以下顺序继续：

1. 用两个 `max_tokens=1` 的顺序请求缩短路径，在每个 decoder layer 边界记录
   hidden state/logits 的 finite、min/max，确定第二请求第一次出错的层；
2. 在首请求前后计算只读的参数/buffer fingerprint，排查 weight 或常驻 buffer 被
   in-place 修改；
3. 重点检查 sparse indexer、compressor、top-k indices buffer、request index、
   block table 和 metadata 的请求间 reset，而不是继续盲目扩大 KV zero 范围；
4. 在找到第一个非有限张量后，再对照 upstream CUDA/XPU 路径核查 shape、stride、
   slot mapping 和生命周期；
5. 修复后重复运行连续请求、logprobs、多轮对话、并发与长上下文测试，再做 NVIDIA
   upstream 同一 W8A8 checkpoint 的准确率对比。

性能优化应放在数值正确性之后。当前 reference fallback 很多，即使连续请求修好，
还需要吞吐、延迟、显存水位和长时间稳定性验收。
