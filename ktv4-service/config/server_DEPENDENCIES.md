# 实际运行代码与依赖位置

项目根：`/mnt/nvme0n1/ktv4`。

启动顺序：`start.sh` → `app/service.py` → `app/launch_workers.sh` → `app/worker.py` → 权重加载及 `app/runtime.py` 初始化 → `app/launch_api.sh` → vLLM API → `app/executor.py` → 四卡worker。问答入口为 `chat.sh` → `app/chat.py`。

| 功能 | 当前源码 |
|---|---|
| 模型结构 | `app/official_reference_model.py` |
| 910A兼容数学、模型构造和权重加载 | `app/legacy_reference.py` |
| CPU路由专家权重与prefill | `app/cpu_experts.py` |
| CPU路由专家decode | `app/fast_cpu_experts.py` |
| 共享专家融合与权重压缩 | `app/fast_moe.py` |
| RMSNorm、RoPE、CPU Sinkhorn | `app/fast_small_ops.py` |
| 注意力和索引 | `app/fast_attention.py`、`app/legacy_indices.py` |
| KV按需增长 | `app/dynamic_kv_cache.py` |
| 前向、采样和EOS相关输入 | `app/runtime.py` |
| vLLM API、调度、tokenizer | `vendor/vllm/vllm/` |
| 实际Ascend插件 | `env210/lib/python3.11/site-packages/vllm_ascend/` |
| PyTorch与CPU INT8矩阵乘 | `env210/lib/python3.11/site-packages/torch/` |
| torch_npu算子和NPU接入 | `env210/lib/python3.11/site-packages/torch_npu/` |
| CANN | `/usr/local/Ascend/cann-9.1.0/` |
| 驱动 | `/usr/local/Ascend/driver/` |
| ATB环境入口 | `/usr/local/Ascend/nnal/atb/set_env.sh` |

保留的优化：CPU专家INT8权重与FP32激活、单token专家计算优化、共享专家投影融合、共享权重存储压缩、融合RMSNorm与RoPE、CPU Sinkhorn、直接缓存注意力、稳定decode索引形状。CPU线程数32。模型计算保留当前已验证的FP16/FP32兼容路径，MTP未启用。

所有已安装包版本保存在 `config/packages.json` 和 `requirements.lock.txt`。vLLM及vLLM-Ascend的上游地址、提交号与修改补丁保存在 `config/`。这些库的运行代码和原生算子源码保留，历史测试、benchmark、文档生成物和构建缓存已清理。

`app/runtime.py` 只保留正式推理路径，没有profiling哨兵、临时基准测试或每token日志。`app/worker.py` 不再逐请求热加载代码，代码变更需要重启服务。
