#include "gpu_macro.h"


void NAME_VAR(lyra2v2_cpu_init_VAR)(int thr_id, uint32_t threads,uint64_t *hash);
void NAME_VAR(lyra2v2_cpu_hash_32_VAR)(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t tpb);
int  NAME_VAR(LYRA2_gpu)(void *K, size_t kLen, const void *pwd, size_t pwdlen, const void *salt, size_t saltlen, size_t timeCost, size_t nRows, size_t nCols);

