#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/mnt/nvme0n1/ktv4/vllm-host/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/mnt/nvme0n1/ktv4/vllm-host/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
