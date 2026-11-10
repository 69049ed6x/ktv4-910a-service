
#ifndef HEADER_ACLRTLAUNCH_BGMV_EXPAND_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_BGMV_EXPAND_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_bgmv_expand_bfloat16_t(uint32_t numBlocks, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim);

inline uint32_t bgmv_expand_bfloat16_t(uint32_t numBlocks, void* hold, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    (void)hold;
    return aclrtlaunch_bgmv_expand_bfloat16_t(numBlocks, stream, x, weight, indices, indicesSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_BGMV_EXPAND_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_BGMV_EXPAND_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_bgmv_expand_half(uint32_t numBlocks, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim);

inline uint32_t bgmv_expand_half(uint32_t numBlocks, void* hold, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    (void)hold;
    return aclrtlaunch_bgmv_expand_half(numBlocks, stream, x, weight, indices, indicesSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_BGMV_SHRINK_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_BGMV_SHRINK_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_bgmv_shrink_bfloat16_t(uint32_t numBlocks, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale);

inline uint32_t bgmv_shrink_bfloat16_t(uint32_t numBlocks, void* hold, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    (void)hold;
    return aclrtlaunch_bgmv_shrink_bfloat16_t(numBlocks, stream, x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_BGMV_SHRINK_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_BGMV_SHRINK_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_bgmv_shrink_half(uint32_t numBlocks, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale);

inline uint32_t bgmv_shrink_half(uint32_t numBlocks, void* hold, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    (void)hold;
    return aclrtlaunch_bgmv_shrink_half(numBlocks, stream, x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_GET_MASKED_INPUT_AND_MASK_KERNEL_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_GET_MASKED_INPUT_AND_MASK_KERNEL_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_get_masked_input_and_mask_kernel(uint32_t numBlocks, void* stream, void* input, void* masked_input, void* mask_out, const int64_t org_vocab_start_index, const int64_t org_vocab_end_index, const int64_t num_org_vocab_padding, const int64_t added_vocab_start_index, const int64_t added_vocab_end_index, const int64_t size, const uint32_t loop_cnt, const uint32_t aiv_num);

inline uint32_t get_masked_input_and_mask_kernel(uint32_t numBlocks, void* hold, void* stream, void* input, void* masked_input, void* mask_out, const int64_t org_vocab_start_index, const int64_t org_vocab_end_index, const int64_t num_org_vocab_padding, const int64_t added_vocab_start_index, const int64_t added_vocab_end_index, const int64_t size, const uint32_t loop_cnt, const uint32_t aiv_num)
{
    (void)hold;
    return aclrtlaunch_get_masked_input_and_mask_kernel(numBlocks, stream, input, masked_input, mask_out, org_vocab_start_index, org_vocab_end_index, num_org_vocab_padding, added_vocab_start_index, added_vocab_end_index, size, loop_cnt, aiv_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_SGMV_EXPAND_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SGMV_EXPAND_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_sgmv_expand_bfloat16_t(uint32_t numBlocks, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim);

inline uint32_t sgmv_expand_bfloat16_t(uint32_t numBlocks, void* hold, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    (void)hold;
    return aclrtlaunch_sgmv_expand_bfloat16_t(numBlocks, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_SGMV_EXPAND_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SGMV_EXPAND_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_sgmv_expand_half(uint32_t numBlocks, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim);

inline uint32_t sgmv_expand_half(uint32_t numBlocks, void* hold, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    (void)hold;
    return aclrtlaunch_sgmv_expand_half(numBlocks, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_SGMV_SHRINK_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SGMV_SHRINK_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_sgmv_shrink_bfloat16_t(uint32_t numBlocks, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale);

inline uint32_t sgmv_shrink_bfloat16_t(uint32_t numBlocks, void* hold, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    (void)hold;
    return aclrtlaunch_sgmv_shrink_bfloat16_t(numBlocks, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_SGMV_SHRINK_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SGMV_SHRINK_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_sgmv_shrink_half(uint32_t numBlocks, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale);

inline uint32_t sgmv_shrink_half(uint32_t numBlocks, void* hold, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    (void)hold;
    return aclrtlaunch_sgmv_shrink_half(numBlocks, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_MLA_PREPROCESS_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_MLA_PREPROCESS_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_mla_preprocess(uint32_t numBlocks, void* stream, void* hiddenState, void* quantScale1, void* quantOffset1, void* wdqkv, void* bias1, void* gamma2, void* beta2, void* quantScale2, void* quantOffset2, void* gamma3, void* sin1, void* cos1, void* sin2, void* cos2, void* keycache, void* slotMapping, void* wuq, void* bias2, void* wuk, void* descale1, void* descale2, void* ctkvScale, void* qnopeScale, void* q, void* keycacheOut, void* q2, void* keycacheOut2, void* innerOut, void* workspace, void* tiling);

inline uint32_t mla_preprocess(uint32_t numBlocks, void* hold, void* stream, void* hiddenState, void* quantScale1, void* quantOffset1, void* wdqkv, void* bias1, void* gamma2, void* beta2, void* quantScale2, void* quantOffset2, void* gamma3, void* sin1, void* cos1, void* sin2, void* cos2, void* keycache, void* slotMapping, void* wuq, void* bias2, void* wuk, void* descale1, void* descale2, void* ctkvScale, void* qnopeScale, void* q, void* keycacheOut, void* q2, void* keycacheOut2, void* innerOut, void* workspace, void* tiling)
{
    (void)hold;
    return aclrtlaunch_mla_preprocess(numBlocks, stream, hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2, quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq, bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2, innerOut, workspace, tiling);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_BATCH_MATMUL_TRANSPOSE_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_BATCH_MATMUL_TRANSPOSE_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_batch_matmul_transpose(uint32_t numBlocks, void* stream, void* gm_a, void* gm_b, void* gm_c, void* gm_tiling_data);

inline uint32_t batch_matmul_transpose(uint32_t numBlocks, void* hold, void* stream, void* gm_a, void* gm_b, void* gm_c, void* gm_tiling_data)
{
    (void)hold;
    return aclrtlaunch_batch_matmul_transpose(numBlocks, stream, gm_a, gm_b, gm_c, gm_tiling_data);
}

#endif
