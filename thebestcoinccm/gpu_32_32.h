#pragma once


void lyra2v2_cpu_init_VAR_32_32(int thr_id, uint32_t threads,uint64_t *hash);
void lyra2v2_cpu_hash_32_VAR_32_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t tpb);
