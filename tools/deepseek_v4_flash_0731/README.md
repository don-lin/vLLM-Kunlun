# DeepSeek V4 Flash 0731 / P800 工具

这些脚本只面向本仓库实现过的实验配置：8 张昆仑芯 P800、TP=8、
compressed-tensors W8A8、BF16 KV、32K、eager。它们不会把当前尚未通过的
连续请求测试包装成“部署成功”。

常用顺序：

```bash
export DSV4_DEPLOY_ROOT=/root/vllm_p800/dsv4
export DSV4_BASE_ENV=/root/miniconda3/envs/vllm025

tools/deepseek_v4_flash_0731/bootstrap_env.sh
tools/deepseek_v4_flash_0731/download_model.sh
tools/deepseek_v4_flash_0731/convert_model.sh
tools/deepseek_v4_flash_0731/preflight.sh
tools/deepseek_v4_flash_0731/start_server.sh
tools/deepseek_v4_flash_0731/smoke_test.py --check-logprobs
```

所有路径都可以通过 `common.sh` 中列出的 `DSV4_*` 环境变量覆盖。环境、模型、
日志和 PID 默认都放在 `DSV4_DEPLOY_ROOT` 下，便于整体删除。不要在 P800 上用
普通 `pip install torch` 替换厂商提供的 `torch_xmlir`/`torch_plugin` 运行栈。
