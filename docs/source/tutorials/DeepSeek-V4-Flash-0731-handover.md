# 当前 P800 机器接手与调试经验

本文面向接手当前远端机器的工程师，记录 2026-08-16 的现场状态。PID、显存和服务
状态会变化，接手后应以命令实时输出为准。文档不保存密码；连接凭据应通过团队的
安全渠道获取。

## 1. 如何连接

当前访问需要经过跳板机，目标 SSH 端口也是 36000：

```bash
ssh -J root@donglinchen-any67.devcloud.woa.com:36000 \
  -p 36000 root@28.175.246.42
```

若需要频繁操作，可在本机 SSH config 中配置 ProxyJump 或 ControlMaster。曾使用的
临时 control socket 是 `/tmp/dsv4-p800-jump-control`，它位于操作端本机，不是远端
服务依赖。通过某些网关直接访问目标 IP 的 HTTP 会得到 ACL 403，这与 vLLM
`/health` 无关；在目标机本地用 `127.0.0.1:8000` 判断服务状态。

## 2. 当前目录和环境

| 内容 | 路径 |
|---|---|
| 工作根目录 | `/root/vllm_p800/dsv4` |
| 独立 Conda 环境 | `/root/vllm_p800/dsv4/conda` |
| 远端 vLLM-Kunlun 源码 | `/root/vllm_p800/dsv4/vLLM-Kunlun` |
| upstream vLLM 0.25.1 | `/root/vllm_p800/dsv4/upstream-vllm-0.25.1` |
| 官方原始模型 | `/root/donlin_model/DeepSeek-V4-Flash-0731` |
| 转换后 W8A8 | `/root/donlin_model/DeepSeek-V4-Flash-0731-W8A8` |
| 当前启动脚本 | `/root/vllm_p800/dsv4/serve-deepseek-v4-p800.sh` |
| 服务日志 | `/root/vllm_p800/dsv4/server.log` |
| PID 文件 | `/root/vllm_p800/dsv4/server.pid` |

模型暂时在 `/root/donlin_model`，没有和独立环境放在同一目录。不要在调试时误删
官方原始模型：全量转换耗时且需要重新占用约 286 GiB 输出空间。

环境参考版本：Python 3.10.20、`torch 2.5.1+cu118`（昆仑定制）、
`torch_plugin 0.1.0`、`torch_xray 2.0.3`、`triton 3.0.0+b2cde523`、vLLM
`0.25.1+empty`、vLLM-Kunlun `0.25.1` editable、compressed-tensors 0.17.0、
transformers 5.15.0、safetensors 0.8.0、openai 3.1.0。

## 3. 接手后的第一轮只读检查

```bash
xpu-smi
cat /root/vllm_p800/dsv4/server.pid
ps -ef | grep -E 'vllm|EngineCore' | grep -v grep
curl -i http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
tail -n 200 /root/vllm_p800/dsv4/server.log
```

2026-08-16 15:50 CST 的最后一次检查中，API server PID 为 162850、EngineCore PID
为 163060；八张卡各使用 90138 MiB，`/health=200`。这些数字只用于识别现场，
不应写死在启停脚本。
日志中八个 worker 都出现过：

```text
KVBlockZeroer registered 62 tensors from 3 groups (specs=['MLAAttentionSpec'])
```

这表示强制 zero 的实验确实进入了 worker，不表示它修好了生成。

## 4. 当前服务与本地仓库的差异

当前远端运行实例带有一次诊断实验：`VLLM_KUNLUN_FORCE_KV_ZERO=1`，远端源码也曾
临时加入“DeepSeek MLA 一律要求 zero”的 property patch。A/B 已证明它不能修复
连续第二请求，因此本地工作分支没有保留这个未提交实验，只保留了与 vLLM 0.25.1
构造 API 匹配的通用 KVBlockZeroer eager 实现。

接手者若要获得可复现基线，应先同步当前 Git 分支到远端、确认 `git status` 干净，
再用新工具脚本重启；不要把当前进程内的 force-zero 状态误认为最终设计。

## 5. 安全重启方法

若远端还没有本次新增工具，先同步仓库，再设置现场模型路径：

```bash
cd /root/vllm_p800/dsv4/vLLM-Kunlun
export DSV4_DEPLOY_ROOT=/root/vllm_p800/dsv4
export DSV4_SOURCE_MODEL=/root/donlin_model/DeepSeek-V4-Flash-0731
export DSV4_MODEL_DIR=/root/donlin_model/DeepSeek-V4-Flash-0731-W8A8
```

对于由新工具启动的服务：

```bash
tools/deepseek_v4_flash_0731/stop_server.sh
xpu-smi
tools/deepseek_v4_flash_0731/start_server.sh
```

当前旧进程不是由新 `setsid` 脚本创建的。第一次迁移时应读取 PID/命令行，确认它们
确实属于这个 DeepSeek 模型，然后终止 API server 及其 EngineCore/worker 子进程，
再用 `ps`、8000 端口和 `xpu-smi` 三重确认已清空。曾经因 stale PID 同时留下三个
API server，所以“kill PID 文件里的一个数字”不够。不要把宽泛的 `pkill python`
写进长期脚本；新 `stop_server.sh` 用独立进程组和命令行校验规避这个问题。

## 6. 必须复现的已知故障

服务启动后不要只发一个请求。运行：

```bash
cd /root/vllm_p800/dsv4/vLLM-Kunlun
export DSV4_DEPLOY_ROOT=/root/vllm_p800/dsv4
export DSV4_MODEL_DIR=/root/donlin_model/DeepSeek-V4-Flash-0731-W8A8

tools/deepseek_v4_flash_0731/smoke_test.py
tools/deepseek_v4_flash_0731/smoke_test.py --check-logprobs
```

当前典型现象是：冷启动后的第一个请求正确；第二个独立请求返回连续 token id 0，
解码成重复 `<｜begin▁of▁sentence｜>`，或者 completion 为空；请求 logprobs 时服务因
`Out of range float values are not JSON compliant` 返回 400。

每次改动后的最小验收应是：冷启动、连续三个确定性请求、再重复一轮、logprobs，
最后才是并发和长上下文。仅 `/health=200` 只能证明进程活着；仅首请求正确会漏掉
当前最重要的错误。

## 7. 已排除的方向与有价值的经验

### 权重转换确实曾经有错，但已修复

官方 dense 权重是 F8_E4M3，scale 在 safetensors 中是 F8_E8M0 原始 byte。旧版把
115–121 当数值，转换后最大 scale 约 419.78，首请求就会 NaN/token 0。按
`2^(byte-127)` 解码并全量重转后，scale 最大约 0.1348425，首请求恢复正确。
因此接手者首先运行 checkpoint validator；不要在一个旧的 W8A8 目录上继续调 runtime。

### Async scheduling 不是唯一原因

加入 `--no-async-scheduling` 后，连续请求错误仍能复现。当前启动脚本保留关闭状态以
减少变量，但不要把调度开关当作已定位根因。

### 盲目扩大 KV zero 没有解决问题

修正 0.25.1 的真实 `KVBlockZeroer.__init__` 入口后，八个 worker 都注册了 62 个
MLA tensor；即使强制每次 zero，第二请求仍失败。继续调试时应先找到第一个非有限
中间张量，而不是继续扩大清零范围。

### 0.15.1 的经验不能原样套用

原来 Qwen 能在 vLLM-Kunlun 0.15.1 跑通，只说明驱动和 P800 基础栈可用。0.25.1
的 cache config、worker 构造入口、scheduler 和 OOT platform 接口不同。特别不要
照搬 `VLLM_USE_V1=1`、旧 KV zero method 名或旧 block-table 假设。

### 依赖环境要克隆，不要重新求解

当前独立环境来自：

```bash
conda create -y -p /root/vllm_p800/dsv4/conda \
  --clone /root/miniconda3/envs/vllm025
```

然后用 `VLLM_TARGET_DEVICE=empty` editable 安装 upstream v0.25.1，再 editable 安装
插件。公共 pip resolver 很可能替换定制 torch；任何依赖调整前先保存 `pip freeze`。

## 8. 推荐的下一步调试方法

1. 用两个顺序的 `max_tokens=1` 请求复现，降低日志量。
2. 在每个 decoder layer 出口记录 `isfinite/all`、min/max；先定位第二请求首次变坏
   的层，再进入该层，不要一开始全模型逐算子打日志。
3. 请求前后给 parameters 和 persistent buffers 做轻量 fingerprint，查 in-place
   weight/buffer mutation。
4. 检查 sparse indexer、compressor、top-k buffer、request index、slot mapping、
   block table 和 metadata 是否按请求重置，尤其关注 reference fallback 与 upstream
   kernel 对 buffer 生命周期的不同假设。
5. 找到第一个异常 tensor 后，对照 upstream vLLM 0.25.1 的 CUDA/XPU shape、stride、
   dtype 和有效行范围。
6. 修复后运行 `smoke_test.py`、logprobs、多轮、并发、32K，再与 NVIDIA upstream
   运行同一 W8A8 checkpoint 的贪心输出/评测集对比。

采集现场信息可运行：

```bash
tools/deepseek_v4_flash_0731/collect_diagnostics.sh
```

它会保存 xpu-smi、pip freeze、Git 状态、进程、模型接口和最近日志，不会主动打包
模型或写入密码。
