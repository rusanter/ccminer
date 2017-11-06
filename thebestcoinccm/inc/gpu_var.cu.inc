

int NAME_VAR(LYRA2_gpu)(void * kOut, size_t kLen, const void * pwd, size_t pwdlen, const void * salt, size_t saltlen, size_t timeCost, size_t nRows, size_t nCols)
{
    if (!((kLen     == 32) &&
          (pwdlen   == 32) &&
          (saltlen  == 32) &&
          (timeCost == Tcost) &&
          (nRows    == Nrow) &&
          (nCols    == Ncol)))
    {
        printf("Error in func '%s' at line %i : bad params\n", __FUNCTION__, __LINE__);
        fprintf(stderr, "Error in func '%s' at line %i : bad params\n", __FUNCTION__, __LINE__);
        return -1;
    }

    uint32_t startNounce = 0;
    unsigned threads = 1024;
    unsigned tpb = 32;

    CudaAlloc alloc_gpu_out(gpu_out, kLen * threads, __FUNCTION__, __LINE__);
    if (!alloc_gpu_out)
        return -1;
    CudaCheck(cudaMemset(gpu_out, 0, kLen * threads), __FUNCTION__, __LINE__);

    CudaAlloc alloc_gpu_matrix(gpu_matrix, Nrow * Ncol * 12 * sizeof(uint64_t) * threads, __FUNCTION__, __LINE__);
    if (!alloc_gpu_matrix)
        return -1;
    CudaCheck(cudaMemset(gpu_matrix, 0, Nrow * Ncol * 12 * sizeof(uint64_t) * threads), __FUNCTION__, __LINE__);

    NAME_VAR(lyra2v2_cpu_init_VAR)(0, threads, gpu_matrix);
    NAME_VAR(lyra2v2_cpu_hash_32_VAR)(0, threads, startNounce, gpu_out, tpb);
    unsigned out_parts = kLen / sizeof(uint64_t);
    for (unsigned part = 0; part < out_parts; part++)
        CudaCheck(cudaMemcpy(((uint64_t *)kOut) + part, gpu_out + part * threads, sizeof(uint64_t), cudaMemcpyDeviceToHost), __FUNCTION__, __LINE__);
    cudaDeviceSynchronize();

    return 0;
}
