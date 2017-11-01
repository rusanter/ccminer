extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "lyra2_params.h"


static _ALIGN(64) uint64_t *d_hash[MAX_GPUS];
static  uint64_t *d_hash2[MAX_GPUS];

extern void blakeKeccak256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash);
extern void blake256_cpu_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash);
extern void Keccak256_cpu_hash_32(int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);

extern void keccak256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
extern void keccak256_cpu_init(int thr_id, uint32_t threads);

extern void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
extern void skein256_cpu_init(int thr_id, uint32_t threads);

extern void skeinCube256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);


extern void lyra2v2_cpu_init_VAR_32_32(int thr_id, uint32_t threads,uint64_t *hash);
extern void lyra2v2_cpu_hash_32_VAR_32_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t tpb);

extern void bmw256_cpu_init(int thr_id, uint32_t threads);
extern void bmw256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resultnonces, uint32_t target);

extern void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash);

extern "C" int LYRA2(void *K, uint64_t kLen, const void *pwd, uint64_t pwdlen, const void *salt, uint64_t saltlen, uint64_t timeCost, uint64_t nRows, uint64_t nCols);

/**
 * CPU version of TheBestCoin algorithm
 */
extern "C" void thebestcoinccm_hash(void *state, const void *input)
{
	sph_blake256_context      ctx_blake;
	sph_keccak256_context     ctx_keccak;
	sph_skein256_context      ctx_skein;
	sph_bmw256_context        ctx_bmw;
	sph_cubehash256_context   ctx_cube;

	uint32_t hashA[8], hashB[8];

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashB, 32);
	sph_cubehash256_close(&ctx_cube, hashA);


	LYRA2(hashB, 32, hashA, 32, hashA, 32, LYRA2_TCOST, LYRA2_ROWS, LYRA2_COLS);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashB, 32);
	sph_skein256_close(&ctx_skein, hashA);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 32);
	sph_cubehash256_close(&ctx_cube, hashB);


	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 32);
	sph_bmw256_close(&ctx_bmw, hashA);

	memcpy(state, hashA, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_thebestcoinccm(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t intensity = 256 * 256 * 8;
	uint32_t tpb = 8;
//	bool mergeblakekeccak = false;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device_map[thr_id]);

	// calculate intensity depending of algo params, may not work for some params and GPUs
	unsigned int vram = 1024 * 1024 * 1536; // 1,5GB VRAM
	unsigned int msize = BLOCK_LEN_BYTES * LYRA2_COLS * LYRA2_ROWS; // Matrix size
	intensity = (vram / msize);

	// Values of tpb and intensity can be changed for specific video card to tune performance
	//if (strstr(props.name, "980 Ti"))
	//{
	//	tpb = 10;
	//	intensity = 256 * 256 * 18;
	//}

//    intensity = 1024 * 1024;
//    tpb = 13;
//    intensity = 128 * 1024;
//    tpb = 29;
    intensity = 16 * 1024;
    tpb = 8;

	uint32_t throughput = device_intensity(device_map[thr_id], __func__, intensity);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00ff;
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (!opt_cpumining) cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}
		//keccak256_cpu_init(thr_id,throughput);
		skein256_cpu_init(thr_id, throughput);
		bmw256_cpu_init(thr_id, throughput);

		applog(LOG_INFO, "GPU #%d: lyra2 params set to R = %d, C = %d, T = %d", thr_id, LYRA2_ROWS, LYRA2_COLS, LYRA2_TCOST);
		applog(LOG_INFO, "GPU #%d: allocating %d MB memory for DMatrix (%d B per thread)", thr_id, ROW_LEN_BYTES * LYRA2_ROWS * throughput / 1048576, ROW_LEN_BYTES * LYRA2_ROWS);
		applog(LOG_INFO, "GPU #%d: throughput is set to %d, tpb = %d", thr_id, throughput, tpb);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash2[thr_id], ROW_LEN_BYTES * LYRA2_ROWS * throughput)); // todo is d_hash2 used now?
		lyra2v2_cpu_init_VAR_32_32(thr_id, throughput, d_hash2[thr_id]);
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint32_t) * throughput));
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	blake256_cpu_setBlock_80(pdata);

	do {
		uint32_t foundNonce[2] = { 0, 0 };

		// Separate calculation may be faster on some GPUs
//		if (mergeblakekeccak)
//		{
			blakeKeccak256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);

/*		}
		else
		{
			blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
			keccak256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);
		}
*/

		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);

		lyra2v2_cpu_hash_32_VAR_32_32(thr_id, throughput, pdata[19], d_hash[thr_id], tpb);

		cudaError_t cer = cudaGetLastError();
		if (cer != cudaSuccess) {
			applog(LOG_INFO, "Cuda error (thebestcoinccm_cpu_hash_32): %s", cudaGetErrorString(cer));
			scan_abort_flag = true;
			break;
		}

		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);
		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);
		bmw256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], foundNonce, ptarget[7]);

		if (foundNonce[0] != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce[0]);
			thebestcoinccm_hash(vhash64, endiandata);
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (foundNonce[1] != 0xffffffff)
				{
					pdata[21] = foundNonce[1];
					res++;
					if (opt_benchmark)  applog(LOG_INFO, "GPU #%d Found second nounce %08x", thr_id, foundNonce[1], vhash64[7], Htarg);
				}
				pdata[19] = foundNonce[0];
				if (opt_benchmark) applog(LOG_INFO, "GPU #%d Found nounce % 08x", thr_id, foundNonce[0], vhash64[7], Htarg);
				return res;
			}
			else
			{
				if (vhash64[7] > Htarg) // don't show message if it is equal but fails fulltest
					applog(LOG_WARNING, "GPU #%d: result does not validate on CPU!", thr_id);
			}
		}

		pdata[19] += throughput;

	} while (!scan_abort_flag && !work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}
