# 在 8 卡 P800 上部署 DeepSeek-V4-Flash-0731 W8A8

本文给出从空白机器开始的可复现流程。这里的“空白”是指没有本仓库、模型和
独立部署环境，但机器已经安装可工作的 P800 驱动、XPU Runtime、厂商 PyTorch
以及 Conda。P800 的 `torch_xmlir`、`torch_plugin` 和定制 PyTorch 不是 PyPI 上的
普通 CUDA PyTorch，本文不能替代厂商驱动和基础镜像的安装手册。

:::{warning}
截至 2026-08-16，这个移植已通过 8 卡启动、完整权重加载、`/health`、模型枚举、
30 个顺序 logprobs 请求、多轮和 2 并发冒烟。长上下文、长期 soak、性能和准确率
对比尚未完成，因此仍属于实验部署，不应直接作为生产可用结论。
:::

## 1. 已验证的配置边界

| 项目 | 值 |
|---|---|
| 加速卡 | 昆仑芯 P800 × 8（每卡 96 GiB） |
| vLLM 基线 | upstream vLLM `v0.25.1` |
| 插件基线 | vLLM-Kunlun `v0.25.1-dev` 及本分支补丁 |
| 并行方式 | 单机 TP=8，`mp` executor |
| 权重 | 官方混合 MXFP4/block-FP8 先离线转换为 compressed-tensors W8A8 |
| 激活 / KV | BF16 / BF16 |
| 上下文 | 最大 32768 |
| 执行模式 | eager，关闭 prefix cache，当前保守配置关闭 async scheduling |

当前没有实现或验证：直接加载官方 MXFP4、DSpark/MTP、FP8 KV cache、Kunlun
Graph、多机和 1M 上下文。平台检查会尽量提前拒绝不支持的组合。

## 2. 机器与磁盘准备

建议至少准备 1 TiB 可用空间。实测官方目录约 156 GiB，转换后的 W8A8 目录约
286 GiB，二者合计约 442 GiB；还需要环境、源码、转换临时空间和日志。

先确认八张卡、驱动和 Runtime：

```bash
xpu-smi
```

已验证机器的参考版本是 P800、driver `5.0.21.43`、XPU-RT `5.0.21`。版本不必
机械地完全相同，但跨大版本时应先验证厂商 PyTorch 能正常枚举八张卡。

准备一个已能跑 P800 vLLM 的基础 Conda 环境，并记录其路径。实测基础栈包括：

- Python 3.10.20
- `torch 2.5.1+cu118`（昆仑定制包，不是公开 CUDA wheel）
- `torch_xmlir`、`torch_plugin 0.1.0`、`torch_xray 2.0.3`
- `triton 3.0.0+b2cde523`
- `compressed-tensors 0.17.0`、`transformers 5.15.0`

## 3. 获取本仓库并创建可整体删除的部署目录

本次修改所在仓库和分支：

```bash
mkdir -p /root/vllm_p800/dsv4
git clone --branch v0.25.1-dev \
  git@github.com:don-lin/vLLM-Kunlun.git \
  /root/vllm_p800/dsv4/vLLM-Kunlun
cd /root/vllm_p800/dsv4/vLLM-Kunlun
```

如果目标机不能访问这个 fork，请把当前分支镜像到可访问的 Git 服务；至少应包含
移植报告中列出的五个提交及本文所在的文档/工具提交。

设置统一部署目录和基础环境。vLLM-Kunlun 源码、Conda 环境、上游源码、模型、日志
和 PID 都会进入 `DSV4_DEPLOY_ROOT`，将来可以整体移走或删除：

```bash
export DSV4_DEPLOY_ROOT=/root/vllm_p800/dsv4
export DSV4_BASE_ENV=/root/miniconda3/envs/vllm025

tools/deepseek_v4_flash_0731/bootstrap_env.sh
```

脚本会做两件事：

1. 克隆基础 Conda 环境到 `$DSV4_DEPLOY_ROOT/conda`；
2. 克隆 upstream `vllm-project/vllm` 的 `v0.25.1`，再以 editable 方式安装上游和
   当前 vLLM-Kunlun。

安装使用 `--no-deps` 是有意为之。不要直接运行会重新解析全部依赖的 `pip install
vllm`，否则公共 torch wheel 可能覆盖 P800 运行栈。若基础环境缺 Python 依赖，
应逐项补齐并再次确认 `torch`、`torch_xmlir`、`torch_plugin` 的来源和版本没有变。

## 4. 并发下载官方模型

先确保环境中有 `huggingface_hub`，并设置访问 Hugging Face 所需的 token 或镜像：

```bash
export DSV4_DEPLOY_ROOT=/root/vllm_p800/dsv4
export DSV4_DOWNLOAD_WORKERS=32
tools/deepseek_v4_flash_0731/download_model.sh
```

脚本调用 `snapshot_download(max_workers=32)`，会并发下载多个 shard，网络较快时可把
worker 数提高到 64。中断后重新运行会复用已下载内容。默认模型是
`deepseek-ai/DeepSeek-V4-Flash-0731`，默认目标是
`$DSV4_DEPLOY_ROOT/models/DeepSeek-V4-Flash-0731`。需要固定模型快照时设置
`DSV4_MODEL_REVISION` 为明确的 commit hash。

## 5. 转成 Kunlun W8A8

官方权重不能直接交给当前 P800 路径。转换器按 shard 流式处理，把 MXFP4 expert
权重和 block-FP8 dense 权重反量化后，转换为动态 per-token activation、对称
per-output-channel INT8 weight 的 compressed-tensors 格式：

```bash
export DSV4_DEPLOY_ROOT=/root/vllm_p800/dsv4
tools/deepseek_v4_flash_0731/convert_model.sh
```

输出目录必须不存在或为空，脚本不会静默覆盖已有模型。转换结束后会自动运行
`validate_checkpoint.py`，检查：

- safetensors index 引用的 shard 是否齐全；
- `model_type=deepseek_v4`、`expert_dtype=int8` 和 compressed-tensors recipe；
- 每个 `.weight_scale` 是否为有限、非零数；
- 最大 scale 是否处在合理范围，从而捕获把 UE8M0 原始字节 `115..121` 当普通数值
  的旧转换错误。

本次完整转换的参考结果是 48 个 shard、约 286 GiB，manifest 中
`quantized=35721`、`copied=878`、`mxfp4=35328`、`block_fp8=390`；weight scale
范围约为 `1.7301305e-05 .. 0.1348425`，零和非有限数均为 0。

## 6. 启动服务

先运行检查：

```bash
export DSV4_DEPLOY_ROOT=/root/vllm_p800/dsv4
tools/deepseek_v4_flash_0731/preflight.sh
```

前台启动适合调试：

```bash
tools/deepseek_v4_flash_0731/serve_p800_tp8.sh
```

后台启动会写入 `$DSV4_DEPLOY_ROOT/server.log` 和 `server.pid`，并最多等待 15 分钟
直到 `/health` 返回成功：

```bash
tools/deepseek_v4_flash_0731/start_server.sh
tail -f /root/vllm_p800/dsv4/server.log
```

不要从旧的 0.15.1 启动脚本继承 `VLLM_USE_V1=1`。0.25.1 已采用对应执行路径，
工具会主动清除该变量以及常见调试变量。已验证启动参数等价于：

```text
TP=8, mp, compressed-tensors, BF16 KV, block-size=256,
max-model-len=32768, max-num-seqs=16, eager,
no-prefix-cache, no-async-scheduling
```

## 7. 验收：必须测试连续请求

基础接口检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

随后运行三个相互独立、顺序发送的确定性请求：

```bash
tools/deepseek_v4_flash_0731/smoke_test.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model deepseek-v4-flash-0731
```

再检查 logprobs 路径：

```bash
tools/deepseek_v4_flash_0731/smoke_test.py --check-logprobs --repeat 10
```

脚本会把 HTTP 状态、文本和服务返回的 token ids 打印出来，并在以下任一情况退出
非零：答案不匹配、token ids 全为 0、重复 BOS token、非 200 响应。当前代码在实机
上应稳定通过该测试。建议再循环至少 10 轮，确保总请求数超过 30，然后再做多轮、
并发和长上下文。

## 8. 停止、诊断与清理

```bash
tools/deepseek_v4_flash_0731/collect_diagnostics.sh
tools/deepseek_v4_flash_0731/stop_server.sh
xpu-smi
```

`stop_server.sh` 只停止由 `start_server.sh` 创建且命令行匹配本部署的进程组；PID
不匹配时会拒绝操作。确认没有需要保留的模型和诊断文件后，整个隔离部署可随
`/root/vllm_p800/dsv4` 一起删除。源码仓库是否保留由操作者自行决定。
