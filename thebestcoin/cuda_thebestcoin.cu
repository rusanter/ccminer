#include <stdio.h>
#include <memory.h>
#include "cuda_vector.h"
#include "sm_32_intrinsics.h"

//===============================================================================
// HELPERS
//===============================================================================

/* Threads per block */
#define TPB 512

/* Kernel bounds helper */
#define BOUNDS(threads,subthreads)											\
	uint32_t tpb = TPB / subthreads;										\
	dim3 grid((threads + tpb - 1) / tpb);									\
	dim3 block(tpb, subthreads);

/* Kernel bounds helper */
#define NBOUNDS(N,threads,subthreads)										\
	uint32_t tpb##N = TPB / subthreads;										\
	dim3 grid##N((threads + tpb - 1) / tpb);								\
	dim3 block##N(tpb, subthreads);

/* Thread id helper*/
#define THREAD(threads,subthreads)											\
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;			\
	const uint32_t subthread = threadIdx.y;									\
	if (thread >= threads || subthread >= subthreads) return;

/* Byte type definition */
typedef unsigned char byte;

//===============================================================================
// LYRA2
//===============================================================================

/* Rotr64 function */
#define ROTR64(w,c) ((w >> c) | (w << (64 - c)))

/* Blake2b's G function */
#define G(a,b,c,d)															\
	a = a + b;																\
	d = ROTR64(d ^ a, 32);													\
	c = c + d;																\
	b = ROTR64(b ^ c, 24);													\
	a = a + b;																\
	d = ROTR64(d ^ a, 16);													\
	c = c + d;																\
	b = ROTR64(b ^ c, 63);

/* Lyra2 function */
__device__
inline static void lyra2(unsigned int subthread, uint64_t *sponge) {
	if (subthread < 4) {
		G(
			sponge[subthread],
			sponge[subthread + 4],
			sponge[subthread + 8],
			sponge[subthread + 12]
		);
		__syncthreads();
		G(
			sponge[subthread],
			sponge[subthread + (subthread >= 3 ? 1 : 5)],
			sponge[subthread + (subthread >= 2 ? 6 : 10)],
			sponge[subthread + (subthread >= 1 ? 11 : 15)]
		);
	}
	__syncthreads();
}

//===============================================================================
// PARAMS
//===============================================================================

/* Matrix size */
#define N_COLS 8
#define N_ROWS 8

/* Other params */
#define N_PARAMS 6
#define pwdLen   0x20
#define saltLen  0x20
#define kLen     0x20
#define timeCost 0x01

/* Params array */
__device__ static const int dParams[N_PARAMS] = {
	kLen,
	pwdLen,
	saltLen,
	timeCost,
	N_ROWS,
	N_COLS
};

//===============================================================================
// MATRIX
//===============================================================================

/* Block length: 768 bits (=96 bytes, =12 uint64_t) */
#define BLOCK_LEN_INT64 12

/* Block length, in bytes */
#define BLOCK_LEN_BYTES (BLOCK_LEN_INT64 * 8)

/* 512 bits (=64 bytes, =8 uint64_t) */
#define BLOCK_LEN_BLAKE2_SAFE_INT64 8

/* Same as above, in bytes */
#define BLOCK_LEN_BLAKE2_SAFE_BYTES (BLOCK_LEN_BLAKE2_SAFE_INT64 * 8)

/* Amount of input blocks */
#define BLOCK_INPUT (((saltLen + pwdLen + 6 * sizeof(int)) / BLOCK_LEN_BLAKE2_SAFE_BYTES) + 1)

/* Size of input blocks */
#define BLOCK_INPUT_BYTES (BLOCK_INPUT * BLOCK_LEN_BLAKE2_SAFE_BYTES)

/* Total length of a row: N_COLS blocks */
#define ROW_LEN_INT64 (BLOCK_LEN_INT64 * N_COLS)

/* Number of bytes per row */
#define ROW_LEN_BYTES (ROW_LEN_INT64 * 8)

/* Current thread matrix */
#define MATRIX(dMatrix)															\
	uint64_t *matrix = (dMatrix + (ROW_LEN_INT64 * N_ROWS * thread));			\
	byte *ptrMatrix = (byte*)& matrix[0];

/* Matrixes buffer */
__device__ uint64_t *dMatrixBuffer[MAX_GPUS];

//===============================================================================
// SPONGE
//===============================================================================

/* Sponge state length */
#define SPONGE_STATE_LEN_INT64 16

/* Sponge state size */
#define SPONGE_STATE_LEN_BYTES (SPONGE_STATE_LEN_INT64 * 8)

/* Sponge states buffer */
__device__ uint64_t *dSpongeBuffer[MAX_GPUS];

/* 512 zero bits + Blake2b IV Array */
__device__ static const uint64_t dSpongeState[16] = {
	0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
	0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
	0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
	0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
	0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
	0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

/* Current thread sponge state */
#define SPONGE(dSponge)															\
	uint64_t *sponge = (dSponge + (SPONGE_STATE_LEN_INT64 * thread));

//===============================================================================
// KERNELS
//===============================================================================

//------------------------------------------------------------------------------
//	SPONGE STATE							
//------------------------------------------------------------------------------

/* Sponge State initialization kernel */
__global__
void thebestcoin_gpu_sponge_state(uint32_t threads, uint64_t *dSponge) {
	THREAD(threads, SPONGE_STATE_LEN_INT64);
	SPONGE(dSponge);

	sponge[subthread] = dSpongeState[subthread];
}

/* Sponge State initialization kernel launcher */
__host__
void thebestcoin_cpu_sponge_state(int thr_id, uint32_t threads) {
	BOUNDS(threads, SPONGE_STATE_LEN_INT64);
	thebestcoin_gpu_sponge_state <<<grid, block>>> (threads, dSpongeBuffer[thr_id]);
	cudaDeviceSynchronize();
}

//------------------------------------------------------------------------------
//	PADDING
//------------------------------------------------------------------------------

/* Padding kernel */
__global__
void thebestcoin_gpu_padding(uint32_t threads, uint64_t *dMatrix, uint2 *dOutputHash) {
	THREAD(threads, 8);
	MATRIX(dMatrix);
	
	// First, we clean enough blocks for the password, salt, params and padding
	if (subthread == 0) memset(ptrMatrix, 0, BLOCK_INPUT_BYTES);
	__syncthreads();

	// Prepends the password and salt
	((uint2 *)ptrMatrix)[subthread] = __ldg(&dOutputHash[thread + (subthread % 4) * threads]);
	ptrMatrix += pwdLen + saltLen;

	// Concatenates the basil: every integer passed as parameter, in the order they are provided by the interface
	if (subthread < N_PARAMS)
		((int *)ptrMatrix)[subthread] = dParams[subthread];

	if (subthread > 0) return;
	ptrMatrix += N_PARAMS * sizeof(int);

	// Now comes the padding
	*ptrMatrix = 0x80; //first byte of padding: right after the password

	// Resets the pointer to the start of the memory matrix
	ptrMatrix = (byte*)& matrix[0];

	// Sets the pointer to the correct position: end of incomplete block
	ptrMatrix += BLOCK_INPUT_BYTES - 1;

	// Last byte of padding: at the end of the last incomplete block
	*ptrMatrix ^= 0x01;
}

/* Padding kernel launcher */
__host__
void thebestcoin_cpu_padding(int thr_id, uint32_t threads, uint64_t *dOutputHash) {
	BOUNDS(threads, 8);
	thebestcoin_gpu_padding <<<grid, block>>> (threads, dMatrixBuffer[thr_id], (uint2*)dOutputHash);
	cudaDeviceSynchronize();
}

//------------------------------------------------------------------------------
//	ABSORB INPUT
//------------------------------------------------------------------------------

/* Absorb input kernel */
__global__
void thebestcoin_gpu_absorb(uint32_t threads, uint64_t *dMatrix, uint64_t *dSponge) {
	THREAD(threads, 8);
	MATRIX(dMatrix);
	SPONGE(dSponge);

	int i, b;
	for (b = 0; b < BLOCK_INPUT; b++) {
		// XORs the first BLOCK_LEN_BLAKE2_SAFE_INT64 words of "in" with the current state
		if (subthread < 8) {
			sponge[subthread] ^= matrix[subthread];
		}
		__syncthreads();

		// Applies the transformation f to the sponge's state
#pragma unroll
		for (i = 0; i < 12; i++) {
			lyra2(subthread, sponge);
		}

		matrix += BLOCK_LEN_BLAKE2_SAFE_INT64;
	}
}

/* Absorb input kernel launcher */
__host__
void thebestcoin_cpu_absorb(int thr_id, uint32_t threads) {
	BOUNDS(threads, 8);
	thebestcoin_gpu_absorb <<<grid, block>>> (threads, dMatrixBuffer[thr_id], dSpongeBuffer[thr_id]);
	cudaDeviceSynchronize();
}

//-------------------------------------------------------------------------------
// REDUCERS
//-------------------------------------------------------------------------------

/* Reduce squeeze kernel */
__global__
void thebestcoin_gpu_reduce_squeeze(uint32_t threads, uint64_t *dMatrix, uint64_t *dSponge) {
	THREAD(threads, BLOCK_LEN_INT64);
	MATRIX(dMatrix);
	SPONGE(dSponge);
	
	int i, j;

	// In Lyra2: pointer to M[0][C-1]
	matrix += (N_COLS - 1) * BLOCK_LEN_INT64;

	// M[0][C-1-col] = H.reduced_squeeze()
	for (i = 0; i < N_COLS; i++) {
		matrix[subthread] = sponge[subthread];
		__syncthreads();

		// Applies the reduced-round transformation f to the sponge's state
		lyra2(subthread, sponge);

		// Goes to next block (column) that will receive the squeezed data
		matrix -= BLOCK_LEN_INT64;
	}
}

/* Reduce squeeze kernel launcher */
__host__
void thebestcoin_cpu_reduce_squeeze(int thr_id, uint32_t threads) {
	BOUNDS(threads, BLOCK_LEN_INT64);
	thebestcoin_gpu_reduce_squeeze <<<grid, block>>> (threads, dMatrixBuffer[thr_id], dSpongeBuffer[thr_id]);
	cudaDeviceSynchronize();
}

/* Reduce duplex kernel */
__global__
void thebestcoin_gpu_reduce_duplex(uint32_t threads, uint64_t *dMatrix, uint64_t *dSponge, unsigned int first, unsigned int second) {
	THREAD(threads, BLOCK_LEN_INT64);
	MATRIX(dMatrix);
	SPONGE(dSponge);

	// Row to feed the sponge
	// In Lyra2: pointer to prev
	uint64_t* ptrWordIn = (uint64_t*)& matrix[first * ROW_LEN_INT64];  

	// Row to receive the sponge's output
	// In Lyra2: pointer to row
	uint64_t* ptrWordOut = (uint64_t*)& matrix[second * ROW_LEN_INT64 + (N_COLS - 1) * BLOCK_LEN_INT64];

	for (int i = 0; i < N_COLS; i++) {
		// Absorbing "M[0][col]"
		sponge[subthread] ^= ptrWordIn[subthread];
		__syncthreads();

		// Applies the reduced-round transformation f to the sponge's state
		lyra2(subthread, sponge);

		// M[1][C-1-col] = M[1][col] XOR rand
		ptrWordOut[subthread] = ptrWordIn[subthread] ^ sponge[subthread];
		__syncthreads();

		// Input: next column (i.e., next block in sequence)
		ptrWordIn += BLOCK_LEN_INT64;

		// Output: goes to previous column
		ptrWordOut -= BLOCK_LEN_INT64;
	}
}

/* Reduce duplex kernel launcher */
__host__
void thebestcoin_cpu_reduce_duplex(int thr_id, uint32_t threads, unsigned int first, unsigned int second) {
	BOUNDS(threads, BLOCK_LEN_INT64);
	thebestcoin_gpu_reduce_duplex <<<grid, block>>> (threads, dMatrixBuffer[thr_id], dSpongeBuffer[thr_id], first, second);
	cudaDeviceSynchronize();
}

/* Reduce duplex filling kernel */
__global__
void thebestcoin_gpu_reduce_duplex_filling(uint32_t threads, uint64_t *dMatrix, uint64_t *dSponge, uint64_t prev0, uint64_t prev1, uint64_t row0, uint64_t row1) {
	THREAD(threads, BLOCK_LEN_INT64);
	MATRIX(dMatrix);
	SPONGE(dSponge);

	int i, j;

	// Row used only as input (rowIn0 or M[prev0])
	// In Lyra2: pointer to prev0, the last row ever initialized
	uint64_t* ptrWordIn0 = (uint64_t *)& matrix[prev0 * ROW_LEN_INT64];              

	// Another row used only as input (rowIn1 or M[prev1])
	// In Lyra2: pointer to prev1, the last row ever revisited and updated
	uint64_t* ptrWordIn1 = (uint64_t *)& matrix[prev1 * ROW_LEN_INT64];              

	// Row used as input and to receive output after rotation (rowInOut or M[row1])
	// In Lyra2: pointer to row1, to be revisited and updated
	uint64_t* ptrWordInOut = (uint64_t *)& matrix[row1 * ROW_LEN_INT64];             

	// Row receiving the output (rowOut or M[row0])
	//In Lyra2: pointer to row0, to be initialized
	uint64_t* ptrWordOut = (uint64_t *)& matrix[(row0 * ROW_LEN_INT64) + ((N_COLS - 1) * BLOCK_LEN_INT64)]; 

	for (i = 0; i < N_COLS; i++) {
		//Absorbing "M[row1] [+] M[prev0] [+] M[prev1]"
		sponge[subthread] ^= (ptrWordInOut[subthread] + ptrWordIn0[subthread] + ptrWordIn1[subthread]);
		__syncthreads();

		//Applies the reduced-round transformation f to the sponge's state
		lyra2(subthread, sponge);

		//M[row0][col] = M[prev0][col] XOR rand
		ptrWordOut[subthread] = ptrWordIn0[subthread] ^ sponge[subthread];
		__syncthreads();

		//M[row1][col] = M[row1][col] XOR rot(rand)
		//rot(): right rotation by 'omega' bits (e.g., 1 or more words)
		//we rotate 2 words for compatibility with the SSE implementation
		ptrWordInOut[subthread] ^= sponge[(subthread + 2) % BLOCK_LEN_INT64];
		__syncthreads();

		//Inputs: next column (i.e., next block in sequence)
		ptrWordInOut += BLOCK_LEN_INT64;
		ptrWordIn0 += BLOCK_LEN_INT64;
		ptrWordIn1 += BLOCK_LEN_INT64;

		//Output: goes to previous column
		ptrWordOut -= BLOCK_LEN_INT64;
	}
}

/* Reduce duplex filling kernel launcher */
__host__
void thebestcoin_cpu_reduce_duplex_filling(int thr_id, uint32_t threads, uint64_t prev0, uint64_t prev1, uint64_t row0, uint64_t row1) {
	BOUNDS(threads, BLOCK_LEN_INT64);
	thebestcoin_gpu_reduce_duplex_filling <<<grid, block>>> (threads, dMatrixBuffer[thr_id], dSpongeBuffer[thr_id], prev0, prev1, row0, row1);
	cudaDeviceSynchronize();
}

//===============================================================================
// WANDERING
//===============================================================================

/* Wandering kernel */
__global__
void thebestcoin_gpu_wandering(uint32_t threads, uint64_t *dMatrix, uint64_t *dSponge, uint64_t prev0, uint64_t prev1) {
	THREAD(threads, BLOCK_LEN_INT64);
	MATRIX(dMatrix);
	SPONGE(dSponge);

	// counter
	int i;

	// row0: sequentially written during Setup; randomly picked during Wandering
	uint64_t row0;      
	uint64_t row1;

	// In Lyra2: col0
	uint64_t randomColumn0;

	// In Lyra2: col1
	uint64_t randomColumn1;

	// In Lyra2: pointer to row0
	uint64_t* ptrWordInOut0 = (uint64_t *)& matrix[row0 * ROW_LEN_INT64];

	// In Lyra2: pointer to row1
	uint64_t* ptrWordInOut1 = (uint64_t *)& matrix[row1 * ROW_LEN_INT64];

	// In Lyra2: pointer to prev1
	uint64_t* ptrWordIn1;

	// In Lyra2: pointer to prev0
	uint64_t* ptrWordIn0;

	// Visitation Loop
	for (uint64_t wCont = 0; wCont < timeCost * N_ROWS; wCont++) {
		// Selects a pseudorandom indices row0 and row1
		// ------------------------------------------------------------------------------------------
		// (USE THIS IF window IS A POWER OF 2)
		// row0 = (((uint64_t)stateLocal[0]) & (nRows-1));
		// row1 = (((uint64_t)stateLocal[2]) & (nRows-1));
		// (USE THIS FOR THE "GENERIC" CASE)
		row0 = ((uint64_t)sponge[0]) % N_ROWS;  //row0 = lsw(rand) mod nRows
		row1 = ((uint64_t)sponge[2]) % N_ROWS;  //row1 = lsw(rot(rand)) mod nRows
		// we rotate 2 words for compatibility with the SSE implementation

		// Performs a reduced-round duplexing operation over "M[row0][col] [+] M[row1][col] [+] M[prev0][col0] [+] M[prev1][col1], updating both M[row0] and M[row1]
		// M[row0][col] = M[row0][col] XOR rand;
		// M[row1][col] = M[row1][col] XOR rot(rand)                     rot(): right rotation by 'omega' bits (e.g., 1 or more words)
		for (i = 0; i < N_COLS; i++) {
			// col0 = lsw(rot^2(rand)) mod N_COLS
			// randomColumn0 = ((uint64_t)sponge[4] & (N_COLS-1))*BLOCK_LEN_INT64;           /*(USE THIS IF N_COLS IS A POWER OF 2)*/
			randomColumn0 = ((uint64_t)sponge[4] % N_COLS) * BLOCK_LEN_INT64;              /*(USE THIS FOR THE "GENERIC" CASE)*/
			ptrWordIn0 = (uint64_t *)& matrix[(prev0 * ROW_LEN_INT64) + randomColumn0];

			// col1 = lsw(rot^3(rand)) mod N_COLS
			// randomColumn1 = ((uint64_t)sponge[6] & (N_COLS-1))*BLOCK_LEN_INT64;           /*(USE THIS IF N_COLS IS A POWER OF 2)*/
			randomColumn1 = ((uint64_t)sponge[6] % N_COLS) * BLOCK_LEN_INT64;              /*(USE THIS FOR THE "GENERIC" CASE)*/
			ptrWordIn1 = (uint64_t *)& matrix[(prev1 * ROW_LEN_INT64) + randomColumn1];

			// Absorbing "M[row0] [+] M[row1] [+] M[prev0] [+] M[prev1]"
			sponge[subthread] ^= (ptrWordInOut0[subthread] + ptrWordInOut1[subthread] + ptrWordIn0[subthread] + ptrWordIn1[subthread]);
			__syncthreads();

			// Applies the reduced-round transformation f to the sponge's state
			lyra2(subthread, sponge);

			// M[rowInOut0][col] = M[rowInOut0][col] XOR rand
			ptrWordInOut0[subthread] ^= sponge[subthread];
			__syncthreads();

			// M[rowInOut1][col] = M[rowInOut1][col] XOR rot(rand)
			// rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			// we rotate 2 words for compatibility with the SSE implementation
			ptrWordInOut1[subthread] ^= sponge[(subthread + 2) % BLOCK_LEN_INT64];
			__syncthreads();

			// Goes to next block
			ptrWordInOut0 += BLOCK_LEN_INT64;
			ptrWordInOut1 += BLOCK_LEN_INT64;
		}

		// update prev: they now point to the last rows ever updated
		prev0 = row0;
		prev1 = row1;
	}

	//============================ Wrap-up Phase ===============================//
	// Absorbs one last block of the memory matrix with the full-round sponge
	uint64_t* ptrWordIn = (uint64_t*)& matrix[(row0 * ROW_LEN_INT64) + 0/*randomColumn0*/];

	// absorbs the column picked
	sponge[subthread] ^= ptrWordIn[subthread];
	__syncthreads();

	// Applies the full-round transformation f to the sponge's state
#pragma unroll
	for (i = 0; i < 12; i++){
		lyra2(subthread, sponge);
	}
}

/* Wandering kernel launcher */
__host__
void thebestcoin_cpu_wandering(int thr_id, uint32_t threads, uint64_t prev0, uint64_t prev1) {
	BOUNDS(threads, BLOCK_LEN_INT64);
	thebestcoin_gpu_wandering <<<grid, block>>> (threads, dMatrixBuffer[thr_id], dSpongeBuffer[thr_id], prev0, prev1);
	cudaDeviceSynchronize();
}

//===============================================================================
// OUTPUT
//===============================================================================

/* Save results kernel */
__global__
void thebestcoin_gpu_output(uint32_t threads, uint64_t *dSponge, uint2 *dOutputHash) {
	THREAD(threads, 4);
	SPONGE(dSponge);

	dOutputHash[thread + subthread * threads] = ((uint2*)sponge)[subthread];
}

/* Save results */
__host__
void thebestcoin_cpu_output(int thr_id, uint32_t threads, uint64_t *dOutputHash) {
	BOUNDS(threads, 4);
	thebestcoin_gpu_output <<<grid, block>>> (threads, dSpongeBuffer[thr_id], (uint2*)dOutputHash);
	cudaDeviceSynchronize();
}

//===============================================================================
// INIT/MAIN
//===============================================================================

/* Allocate memory */
__host__
void thebestcoin_cpu_init(int thr_id, uint32_t threads) {
	CUDA_SAFE_CALL(cudaMalloc(&dMatrixBuffer[thr_id], ROW_LEN_BYTES * N_ROWS * threads));
	CUDA_SAFE_CALL(cudaMalloc(&dSpongeBuffer[thr_id], SPONGE_STATE_LEN_BYTES * threads));
}

/* Main function */
__host__ 
void thebestcoin_cpu_hash_32(int thr_id, uint32_t threads, uint64_t *dOutputHash) {
	int64_t gap = 1;     // Modifier to the step, assuming the values 1 or -1
	uint64_t step = 1;   // Visitation step (used during Setup to dictate the sequence in which rows are read)
	uint64_t window = 2; // Visitation window (used to define which rows can be revisited during Setup)
	uint64_t sqrt = 2;   // Square of window (i.e., square(window)), when a window is a square number;
						 // otherwise, sqrt = 2*square(window/2)
	uint64_t row0 = 3;   // row0: sequentially written during Setup; randomly picked during Wandering
	uint64_t prev0 = 2;  // prev0: stores the previous value of row0
	uint64_t row1 = 1;   // row1: revisited during Setup, and then read [and written]; randomly picked during Wandering
	uint64_t prev1 = 0;  // prev1: stores the previous value of row1

	// Setup
	thebestcoin_cpu_sponge_state(thr_id, threads);
	thebestcoin_cpu_padding(thr_id, threads, dOutputHash);
	thebestcoin_cpu_absorb(thr_id, threads);
	thebestcoin_cpu_reduce_squeeze(thr_id, threads);
	thebestcoin_cpu_reduce_duplex(thr_id, threads, 0, 1);
	thebestcoin_cpu_reduce_duplex(thr_id, threads, 1, 2);

	// Filling
	for (row0 = 3; row0 < N_ROWS; row0++) {
		// Performs a reduced-round duplexing operation over "M[row1][col] [+] M[prev0][col] [+] M[prev1][col]", filling M[row0] and updating M[row1]
		// M[row0][N_COLS-1-col] = M[prev0][col] XOR rand;
		// M[row1][col] = M[row1][col] XOR rot(rand)
		//
		// rot(): right rotation by 'omega' bits (e.g., 1 or more words)
		thebestcoin_cpu_reduce_duplex_filling(thr_id, threads, prev0, prev1, row0, row1);

		//Updates the "prev" indices: the rows more recently updated
		prev0 = row0;
		prev1 = row1;

		//updates the value of row1: deterministically picked, with a variable step
		row1 = (row1 + step) & (window - 1);

		//Checks if all rows in the window where visited.
		if (row1 == 0) {
			window *= 2;            //doubles the size of the re-visitation window
			step = sqrt + gap;      //changes the step: approximately doubles its value
			gap = -gap;             //inverts the modifier to the step
			if (gap == -1) {
				sqrt *= 2;          //Doubles sqrt every other iteration
			}
		}
	}

	// Wandering
	thebestcoin_cpu_wandering(thr_id, threads, prev0, prev1);

	// Output
	thebestcoin_cpu_output(thr_id, threads, dOutputHash);
}

  
