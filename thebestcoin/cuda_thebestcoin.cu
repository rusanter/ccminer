

#include <stdio.h>
#include <memory.h>
#include "cuda_vector.h"
#define TPB52 256
#define TPB50 64

 
#define Nrow 8
#define Ncol 8
#define N_ROWS Nrow
#define N_COLS Ncol
#define u64type uint2
#define vectype uint28
#define memshift 3
__device__ uint64_t  *DMatrix;

#define nPARALLEL 1
#define STATESIZE_INT64 16
#define STATESIZE_BYTES (STATESIZE_INT64 * sizeof (uint64_t))
#define RHO 1
#define BLOCK_LEN_INT64 12                                      //Block length: 768 bits (=96 bytes, =12 uint64_t)
#define BLOCK_LEN_BYTES (BLOCK_LEN_INT64 * 8)                   //Block length, in bytes
#define ROW_LEN_INT64 (BLOCK_LEN_INT64 * Ncol)                  //Total length of a row: N_COLS blocks
#define ROW_LEN_BYTES (ROW_LEN_INT64 * 8)                       //Number of bytes per row
//Block length required so Blake2's Initialization Vector (IV) is not overwritten (THIS SHOULD NOT BE MODIFIED)
#define BLOCK_LEN_BLAKE2_SAFE_INT64 8                                   //512 bits (=64 bytes, =8 uint64_t)
#define BLOCK_LEN_BLAKE2_SAFE_BYTES (BLOCK_LEN_BLAKE2_SAFE_INT64 * 8)   //same as above, in bytes

#define pwdlen   0x20
#define saltlen  0x20
#define kLen     0x20
#define timeCost 1
#define nRows    Nrow
#define nCols    Ncol

typedef unsigned char byte;

__device__ uint64_t sizeSlicedRows;

/*Blake2b IV Array*/
__device__ static const uint64_t blake2b_IV[8] =
{
  0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
  0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
  0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
  0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

//////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////       SPONGE       /////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

/*Blake2b's rotation*/
__device__ static inline uint64_t rotr64( const uint64_t w, const unsigned c ){
    return ( w >> c ) | ( w << ( 64 - c ) );
}

/*Blake2b's G function*/
#define G(r,i,a,b,c,d) \
  do { \
    a = a + b; \
    d = rotr64(d ^ a, 32); \
    c = c + d; \
    b = rotr64(b ^ c, 24); \
    a = a + b; \
    d = rotr64(d ^ a, 16); \
    c = c + d; \
    b = rotr64(b ^ c, 63); \
  } while(0)

/*One Round of the Blake2b's compression function*/
#define ROUND_LYRA(r)  \
    G(r,0,v[ 0],v[ 4],v[ 8],v[12]); \
    G(r,1,v[ 1],v[ 5],v[ 9],v[13]); \
    G(r,2,v[ 2],v[ 6],v[10],v[14]); \
    G(r,3,v[ 3],v[ 7],v[11],v[15]); \
    G(r,4,v[ 0],v[ 5],v[10],v[15]); \
    G(r,5,v[ 1],v[ 6],v[11],v[12]); \
    G(r,6,v[ 2],v[ 7],v[ 8],v[13]); \
    G(r,7,v[ 3],v[ 4],v[ 9],v[14]);

/**
 * Execute G function, with all 12 rounds for Blake2 and  BlaMka, and 24 round for half-round BlaMka.
 *
 * @param v     A 1024-bit (16 uint64_t) array to be processed by Blake2b's or BlaMka's G function
 */
__device__ inline static void spongeLyra(uint64_t *v) {
    int i;

#pragma unroll
    for (i = 0; i < 12; i++){
        ROUND_LYRA(i);
    }
}

/**
 * Executes a reduced version of G function with only RHO round
 * @param v     A 1024-bit (16 uint64_t) array to be processed by Blake2b's or BlaMka's G function
 */
__device__ inline static void reducedSpongeLyra(uint64_t *v) {
    int i;

#pragma unroll
    for (i = 0; i < RHO; i++){
        ROUND_LYRA(i);
    }
}

//////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////       ABSORBS      /////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

/**
 * Performs an absorb operation for a single block (BLOCK_LEN_BLAKE2_SAFE_INT64
 * words of type uint64_t), using G function as the internal permutation
 *
 * @param state         The current state of the sponge
 * @param in            The block to be absorbed (BLOCK_LEN_BLAKE2_SAFE_INT64 words)
 */
__device__ inline void absorbBlockBlake2Safe(uint64_t *state, const uint64_t *in) {
    //XORs the first BLOCK_LEN_BLAKE2_SAFE_INT64 words of "in" with the current state
    state[0] ^= in[0];
    state[1] ^= in[1];
    state[2] ^= in[2];
    state[3] ^= in[3];
    state[4] ^= in[4];
    state[5] ^= in[5];
    state[6] ^= in[6];
    state[7] ^= in[7];

    //Applies the transformation f to the sponge's state
    spongeLyra(state);
}

//////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////      REDUCERS      /////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

/**
 * Performs a reduced squeeze operation for a single row, from the highest to
 * the lowest index, using the reduced-round G function as the
 * internal permutation
 *
 * @param state          The current state of the sponge
 * @param rowOut         Row to receive the data squeezed
 */
__device__ void reducedSqueezeRow0(uint64_t* rowOut, uint64_t* state) {
    const int threadNumber = 0;
    uint64_t sliceStart;
    uint64_t stateStart;

    if (threadNumber < (nPARALLEL)) {
        stateStart = threadNumber * STATESIZE_INT64;
        sliceStart = threadNumber * sizeSlicedRows;

        uint64_t* ptrWord = &rowOut[sliceStart + (N_COLS - 1) * BLOCK_LEN_INT64]; //In Lyra2: pointer to M[0][C-1]
        int i, j;
        //M[0][C-1-col] = H.reduced_squeeze()
        for (i = 0; i < N_COLS; i++) {
            for (j = 0; j < BLOCK_LEN_INT64; j++) {
                ptrWord[j] = state[stateStart + j];
            }

            //Goes to next block (column) that will receive the squeezed data
            ptrWord -= BLOCK_LEN_INT64;

            //Applies the reduced-round transformation f to the sponge's state
            reducedSpongeLyra(&state[stateStart]);
        }
    }
}

/**
* Performs a reduced duplex operation for a single row, from the highest to
* the lowest index of its columns, using the reduced-round G function
* as the internal permutation
*
* @param state                 The current state of the sponge
* @param rowIn                 Matrix start (base row)
* @param first                 Index used with rowIn to calculate wich row will feed the sponge
* @param second                Index used with rowIn to calculate wich row will receive the sponge's state
*/
__device__ void reducedDuplexRow1and2(uint64_t *rowIn, uint64_t *state, unsigned int first, unsigned int second) {
	int i, j;

	const int threadNumber = 0;
	uint64_t sliceStart;
	uint64_t stateStart;

	if (threadNumber < (nPARALLEL)) {

		stateStart = threadNumber * STATESIZE_INT64;
		sliceStart = threadNumber * sizeSlicedRows;

		//Row to feed the sponge
		uint64_t* ptrWordIn = (uint64_t*)& rowIn[sliceStart + first * ROW_LEN_INT64];                                          //In Lyra2: pointer to prev
		//Row to receive the sponge's output
		uint64_t* ptrWordOut = (uint64_t*)& rowIn[sliceStart + second * ROW_LEN_INT64 + (N_COLS - 1) * BLOCK_LEN_INT64];       //In Lyra2: pointer to row

		for (i = 0; i < N_COLS; i++) {
			//Absorbing "M[0][col]"
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				state[stateStart + j] ^= (ptrWordIn[j]);
			}

			//Applies the reduced-round transformation f to the sponge's state
			reducedSpongeLyra(&state[stateStart]);

			//M[1][C-1-col] = M[1][col] XOR rand
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordOut[j] = ptrWordIn[j] ^ state[stateStart + j];
			}

			//Input: next column (i.e., next block in sequence)
			ptrWordIn += BLOCK_LEN_INT64;
			//Output: goes to previous column
			ptrWordOut -= BLOCK_LEN_INT64;
		}
	}
}

/**
* Performs an absorb operation of single column from "in", the
* said column being pseudorandomly picked in the range [0, BLOCK_LEN_INT64[,
* using the full-round G function as the internal permutation
*
* @param state                         The current state of the sponge
* @param in    			Matrix start
* @param row0				The row whose column (BLOCK_LEN_INT64 words) should be absorbed
* @param randomColumn0                 The random column to be absorbed
*/
__device__ void absorbRandomColumn(uint64_t *in, uint64_t *state, uint64_t row0, uint64_t randomColumn0) {
	int i;
	const int threadNumber = 0;
	uint64_t sliceStart;
	uint64_t stateStart;

	if (threadNumber < (nPARALLEL)) {
		stateStart = threadNumber * STATESIZE_INT64;
		sliceStart = threadNumber * sizeSlicedRows;

		uint64_t* ptrWordIn = (uint64_t*)& in[sliceStart + (row0 * ROW_LEN_INT64) + randomColumn0];

		//absorbs the column picked
		for (i = 0; i < BLOCK_LEN_INT64; i++) {
			state[stateStart + i] ^= ptrWordIn[i];
		}

		//Applies the full-round transformation f to the sponge's state
		spongeLyra(&state[stateStart]);
	}
}

/**
* Performs a squeeze operation, using G function as the
* internal permutation
*
* @param state          The current state of the sponge
* @param out            Array that will receive the data squeezed
* @param len            The number of bytes to be squeezed into the "out" array
*/
__device__ void squeezeGPU(uint64_t *state, byte *out, unsigned int len) {
	int i;
	int fullBlocks = len / BLOCK_LEN_BYTES;

	const int threadNumber = 0;
	uint64_t stateStart;

	if (threadNumber < (nPARALLEL)) {

		stateStart = threadNumber * STATESIZE_INT64;
		byte *ptr = (byte *)& out[threadNumber * len];

		//Squeezes full blocks
		for (i = 0; i < fullBlocks; i++) {
			memcpy(ptr, &state[stateStart], BLOCK_LEN_BYTES);
			spongeLyra(&state[stateStart]);
			ptr += BLOCK_LEN_BYTES;
		}

		//Squeezes remaining bytes
		memcpy(ptr, &state[stateStart], (len % BLOCK_LEN_BYTES));
	}
}

/**
 * Performs a initial absorb operation
 * Absorbs salt, password and the other parameters
 *
 * @param memMatrixGPU        Matrix start
 * @param stateThreadGPU    The current state of the sponge
 * @param stateIdxGPU          Index of the threads, to be absorbed
 * @param nBlocksInput         The number of blocks to be absorbed
 */
__device__ void absorbInput(uint64_t * memMatrixGPU, uint64_t * stateThreadGPU, uint64_t *stateIdxGPU, uint64_t nBlocksInput) {
    uint64_t *ptrWord;
    uint64_t *threadState;
    const int threadNumber = 0;
    uint64_t kP;
    uint64_t sliceStart;

    if (threadNumber < (nPARALLEL)) {
        sliceStart = threadNumber*sizeSlicedRows;
        threadState = (uint64_t *) & stateThreadGPU[threadNumber * STATESIZE_INT64];

        //Absorbing salt, password and params: this is the only place in which the block length is hard-coded to 512 bits, for compatibility with Blake2b and BlaMka
        ptrWord = (uint64_t *) & memMatrixGPU[sliceStart];              //threadSliceMatrix;
        for (kP = 0; kP < nBlocksInput; kP++) {
            absorbBlockBlake2Safe(threadState, ptrWord);                //absorbs each block of pad(pwd || salt || params)
            ptrWord += BLOCK_LEN_BLAKE2_SAFE_INT64;                     //BLOCK_LEN_BLAKE2_SAFE_INT64;  //goes to next block of pad(pwd || salt || params)
        }
    }
}

/**
* Performs a duplexing operation over
* "M[rowInOut0][col] [+] M[rowInOut1][col] [+] M[rowIn0][col_0] [+] M[rowIn1][col_1]",
* where [+] denotes wordwise addition, ignoring carries between words. The value of
* "col_0" is computed as "lsw(rot^2(rand)) mod N_COLS", and "col_1" as
* "lsw(rot^3(rand)) mod N_COLS", where lsw() means "the least significant word"
* where rot is a right rotation by 'omega' bits (e.g., 1 or more words),
* N_COLS is a system parameter, and "rand" corresponds
* to the sponge's output for each column absorbed.
* The same output is then employed to make
* "M[rowInOut0][col] = M[rowInOut0][col] XOR rand" and
* "M[rowInOut1][col] = M[rowInOut1][col] XOR rot(rand)".
*
* @param memMatrixGPU          Matrix start
* @param stateLocal            The current state of the sponge
* @param prev0			Row used only as input
* @param row0			Row used as input and to receive output
* @param prev1			Another row used only as input
* @param row1			Row used as input and to receive output after rotation
*/
__device__ void reducedDuplexRowWanderingOTM_P1(uint64_t *memMatrixGPU, uint64_t *stateLocal, uint64_t prev0, uint64_t row0, uint64_t row1, uint64_t prev1) {
	const int threadNumber = 0;

	uint64_t randomColumn0; //In Lyra2: col0
	uint64_t randomColumn1; //In Lyra2: col1

	if (threadNumber < (nPARALLEL)) {
		uint64_t* ptrWordInOut0 = (uint64_t *)& memMatrixGPU[row0 * ROW_LEN_INT64]; //In Lyra2: pointer to row0
		uint64_t* ptrWordInOut1 = (uint64_t *)& memMatrixGPU[row1 * ROW_LEN_INT64]; //In Lyra2: pointer to row1

		uint64_t* ptrWordIn1; //In Lyra2: pointer to prev1
		uint64_t* ptrWordIn0; //In Lyra2: pointer to prev0

		int i, j;

		for (i = 0; i < N_COLS; i++) {
			//col0 = lsw(rot^2(rand)) mod N_COLS
			//randomColumn0 = ((uint64_t)stateLocal[4] & (N_COLS-1))*BLOCK_LEN_INT64;           /*(USE THIS IF N_COLS IS A POWER OF 2)*/
			randomColumn0 = ((uint64_t)stateLocal[4] % N_COLS) * BLOCK_LEN_INT64;              /*(USE THIS FOR THE "GENERIC" CASE)*/
			ptrWordIn0 = (uint64_t *)& memMatrixGPU[(prev0 * ROW_LEN_INT64) + randomColumn0];

			//col1 = lsw(rot^3(rand)) mod N_COLS
			//randomColumn1 = ((uint64_t)stateLocal[6] & (N_COLS-1))*BLOCK_LEN_INT64;           /*(USE THIS IF N_COLS IS A POWER OF 2)*/
			randomColumn1 = ((uint64_t)stateLocal[6] % N_COLS) * BLOCK_LEN_INT64;              /*(USE THIS FOR THE "GENERIC" CASE)*/
			ptrWordIn1 = (uint64_t *)& memMatrixGPU[(prev1 * ROW_LEN_INT64) + randomColumn1];

			//Absorbing "M[row0] [+] M[row1] [+] M[prev0] [+] M[prev1]"
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				stateLocal[j] ^= (ptrWordInOut0[j] + ptrWordInOut1[j] + ptrWordIn0[j] + ptrWordIn1[j]);
			}

			//Applies the reduced-round transformation f to the sponge's state
			reducedSpongeLyra(stateLocal);

			//M[rowInOut0][col] = M[rowInOut0][col] XOR rand
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordInOut0[j] ^= stateLocal[j];
			}

			//M[rowInOut1][col] = M[rowInOut1][col] XOR rot(rand)
			//rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			//we rotate 2 words for compatibility with the SSE implementation
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordInOut1[j] ^= stateLocal[(j + 2) % BLOCK_LEN_INT64];
			}

			//Goes to next block
			ptrWordInOut0 += BLOCK_LEN_INT64;
			ptrWordInOut1 += BLOCK_LEN_INT64;

		}
	}
}

/**
* Wandering phase: performs the visitation loop
* Visitation loop chooses pseudo random rows (row0 and row1) based in state content
* And performs a reduced-round duplexing operation over:
* "M[row0][col] [+] M[row1][col] [+] M[prev0][col0] [+] M[prev1][col1]
* Updating both M[row0] and M[row1] using the output to make:
* M[row0][col] = M[row0][col] XOR rand;
* M[row1][col] = M[row1][col] XOR rot(rand)
* Where rot() is a right rotation by 'omega' bits (e.g., 1 or more words)
*
* @param stateLocal         	The current state of the sponge
* @param memMatrixGPU 		Array that will receive the data squeezed
* @param prev0                 Stores the previous value of row0
* @param prev1                 Stores the previous value of row1
*/
__device__ void wanderingPhaseGPU2_P1(uint64_t * memMatrixGPU, uint64_t * stateLocal, uint64_t prev0, uint64_t prev1) {
	uint64_t wCont;     //Time Loop iterator
	uint64_t row0;      //row0: sequentially written during Setup; randomly picked during Wandering
	uint64_t row1;
	const int threadNumber = 0;

	if (threadNumber < (nPARALLEL)) {
		//Visitation Loop
		for (wCont = 0; wCont < timeCost * nRows; wCont++) {
			//Selects a pseudorandom indices row0 and row1
			//------------------------------------------------------------------------------------------
			//(USE THIS IF window IS A POWER OF 2)
			//row0 = (((uint64_t)stateLocal[0]) & (nRows-1));
			//row1 = (((uint64_t)stateLocal[2]) & (nRows-1));
			//(USE THIS FOR THE "GENERIC" CASE)
			row0 = ((uint64_t)stateLocal[0]) % nRows;  //row0 = lsw(rand) mod nRows
			row1 = ((uint64_t)stateLocal[2]) % nRows;  //row1 = lsw(rot(rand)) mod nRows
			//we rotate 2 words for compatibility with the SSE implementation

			//Performs a reduced-round duplexing operation over "M[row0][col] [+] M[row1][col] [+] M[prev0][col0] [+] M[prev1][col1], updating both M[row0] and M[row1]
			//M[row0][col] = M[row0][col] XOR rand;
			//M[row1][col] = M[row1][col] XOR rot(rand)                     rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			reducedDuplexRowWanderingOTM_P1(memMatrixGPU, stateLocal, prev0, row0, row1, prev1);

			//update prev: they now point to the last rows ever updated
			prev0 = row0;
			prev1 = row1;
		}
		//============================ Wrap-up Phase ===============================//
		//Absorbs one last block of the memory matrix with the full-round sponge
		absorbRandomColumn(memMatrixGPU, stateLocal, row0, 0);
	}
}

/**
* Performs a duplexing operation over
* "M[rowInOut][col] [+] M[rowIn0][col] [+] M[rowIn1][col]", where [+] denotes
* wordwise addition, ignoring carries between words, for all values of "col"
* in the [0,N_COLS[ interval. The  output of this operation, "rand", is then
* employed to make
* "M[rowOut][(N_COLS-1)-col] = M[rowIn0][col] XOR rand" and
* "M[rowInOut][col] =  M[rowInOut][col] XOR rot(rand)",
* where rot is a right rotation by 'omega' bits (e.g., 1 or more words)
* and N_COLS is a system parameter.
*
* @param stateLocal            The current state of the sponge
* @param memMatrixGPU          Matrix start
* @param prev0			The last row ever initialized
* @param prev1			The last row ever revisited and updated
* @param row0			Row to be initialized
* @param row1			Row to be revisited and updated
*/
__device__ void reducedDuplexRowFilling2OTM_P1(uint64_t *stateLocal, uint64_t *memMatrixGPU, uint64_t prev0, uint64_t prev1, uint64_t row0, uint64_t row1) {
	int i, j;
	const int threadNumber = 0;

	if (threadNumber < (nPARALLEL)) {

		//Row used only as input (rowIn0 or M[prev0])
		uint64_t* ptrWordIn0 = (uint64_t *)& memMatrixGPU[prev0 * ROW_LEN_INT64];              //In Lyra2: pointer to prev0, the last row ever initialized

		//Another row used only as input (rowIn1 or M[prev1])
		uint64_t* ptrWordIn1 = (uint64_t *)& memMatrixGPU[prev1 * ROW_LEN_INT64];              //In Lyra2: pointer to prev1, the last row ever revisited and updated

		//Row used as input and to receive output after rotation (rowInOut or M[row1])
		uint64_t* ptrWordInOut = (uint64_t *)& memMatrixGPU[row1 * ROW_LEN_INT64];             //In Lyra2: pointer to row1, to be revisited and updated

		//Row receiving the output (rowOut or M[row0])
		uint64_t* ptrWordOut = (uint64_t *)& memMatrixGPU[(row0 * ROW_LEN_INT64) + ((N_COLS - 1) * BLOCK_LEN_INT64)]; //In Lyra2: pointer to row0, to be initialized

		for (i = 0; i < N_COLS; i++) {
			//Absorbing "M[row1] [+] M[prev0] [+] M[prev1]"
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				stateLocal[j] ^= (ptrWordInOut[j] + ptrWordIn0[j] + ptrWordIn1[j]);
			}

			//Applies the reduced-round transformation f to the sponge's state
			reducedSpongeLyra(stateLocal);

			//M[row0][col] = M[prev0][col] XOR rand
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordOut[j] = ptrWordIn0[j] ^ stateLocal[j];
			}

			//M[row1][col] = M[row1][col] XOR rot(rand)
			//rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			//we rotate 2 words for compatibility with the SSE implementation
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordInOut[j] ^= stateLocal[(j + 2) % BLOCK_LEN_INT64];
			}

			//Inputs: next column (i.e., next block in sequence)
			ptrWordInOut += BLOCK_LEN_INT64;
			ptrWordIn0 += BLOCK_LEN_INT64;
			ptrWordIn1 += BLOCK_LEN_INT64;
			//Output: goes to previous column
			ptrWordOut -= BLOCK_LEN_INT64;
		}
	}
}

/**
 * Initializes the Sponge's State. The first 512 bits are set to zeros and the remainder
 * receive Blake2b's IV as per Blake2b's specification. <b>Note:</b> Even though sponges
 * typically have their internal state initialized with zeros, Blake2b's G function
 * has a fixed point: if the internal state and message are both filled with zeros. the
 * resulting permutation will always be a block filled with zeros; this happens because
 * Blake2b does not use the constants originally employed in Blake2 inside its G function,
 * relying on the IV for avoiding possible fixed points.
 *
 * @param state          The 1024-bit array to be initialized
 */
__device__ void initState(uint64_t state[/*16*/]) {

    const int threadNumber = 0;

    if (threadNumber < (nPARALLEL)) {

        const uint64_t start = threadNumber * STATESIZE_INT64;

        //First 512 bis are zeros
        state[start + 0] = 0x0ULL;
        state[start + 1] = 0x0ULL;
        state[start + 2] = 0x0ULL;
        state[start + 3] = 0x0ULL;
        state[start + 4] = 0x0ULL;
        state[start + 5] = 0x0ULL;
        state[start + 6] = 0x0ULL;
        state[start + 7] = 0x0ULL;
        //Remainder BLOCK_LEN_BLAKE2_SAFE_BYTES are reserved to the IV
        state[start + 8] = blake2b_IV[0];
        state[start + 9] = blake2b_IV[1];
        state[start + 10] = blake2b_IV[2];
        state[start + 11] = blake2b_IV[3];
        state[start + 12] = blake2b_IV[4];
        state[start + 13] = blake2b_IV[5];
        state[start + 14] = blake2b_IV[6];
        state[start + 15] = blake2b_IV[7];
    }
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(128, 1)
#endif
void thebestcoin_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

    // Size of each chunk that each thread will work with
    //updates global sizeSlicedRows;
    sizeSlicedRows = (Nrow / nPARALLEL) * ROW_LEN_INT64;

    uint64_t stateLocal[16];

    if (thread < threads)
    {
		const uint32_t ps = (ROW_LEN_INT64 * Nrow * thread);
        const uint64_t stateIdxGPU = 0;                            // TODO get rid of this?
        const uint64_t nBlocksInput = ((saltlen + pwdlen + 6 * sizeof (int)) / BLOCK_LEN_BLAKE2_SAFE_BYTES) + 1; // = 2
        uint64_t *memMatrixGPU = (DMatrix + ps);
        int i;
        byte *ptrByte;

        // 1. Bootstrapping phase

        //======================= Initializing the Sponge State ====================//
        //Sponge state: 16 uint64_t, BLOCK_LEN_INT64 words of them for the bitrate (b) and the remainder for the capacity (c)
        initState(stateLocal);

        //============= Padding (password + salt + params) with 10*1 ===============//
        //OBS.:The memory matrix will temporarily hold the password: not for saving memory,
        //but this ensures that the password copied locally will be overwritten as soon as possible
        ptrByte = (byte*) & memMatrixGPU[0];

        //First, we clean enough blocks for the password, salt, params and padding
        for (i = 0; i < nBlocksInput * BLOCK_LEN_BLAKE2_SAFE_BYTES; i++) {
            ptrByte[i] = (byte) 0;
        }

        //Prepends the password
		((uint2 *)ptrByte)[0] = __ldg(&outputHash[thread + 0 * threads]);
		((uint2 *)ptrByte)[1] = __ldg(&outputHash[thread + 1 * threads]);
		((uint2 *)ptrByte)[2] = __ldg(&outputHash[thread + 2 * threads]);
		((uint2 *)ptrByte)[3] = __ldg(&outputHash[thread + 3 * threads]);
        ptrByte += pwdlen;

        //The indexed salt
		((uint2 *)ptrByte)[0] = __ldg(&outputHash[thread]);
		((uint2 *)ptrByte)[1] = __ldg(&outputHash[thread + threads]);
		((uint2 *)ptrByte)[2] = __ldg(&outputHash[thread + 2 * threads]);
		((uint2 *)ptrByte)[3] = __ldg(&outputHash[thread + 3 * threads]);
		ptrByte += saltlen;

		//Concatenates the basil: every integer passed as parameter, in the order they are provided by the interface
		((int *)ptrByte)[0] = kLen;
		((int *)ptrByte)[1] = pwdlen;
		((int *)ptrByte)[2] = saltlen;
		((int *)ptrByte)[3] = timeCost;
		((int *)ptrByte)[4] = nRows;
		((int *)ptrByte)[5] = nCols;

		//memcpy(&outputHash[thread + 0 * threads], ptrByte, sizeof(uint64_t));
		//memcpy(&outputHash[thread + 1 * threads], ptrByte + 8, sizeof(uint64_t));
		//memcpy(&outputHash[thread + 2 * threads], ptrByte + 16, sizeof(uint64_t));
		//memcpy(&outputHash[thread + 3 * threads], ptrByte + 24, sizeof(uint64_t));
		//return;

		ptrByte += 6 * sizeof(int);

		//Now comes the padding
		*ptrByte = 0x80; //first byte of padding: right after the password

		//resets the pointer to the start of the memory matrix
		ptrByte = (byte*) & memMatrixGPU[0];
		ptrByte += nBlocksInput * BLOCK_LEN_BLAKE2_SAFE_BYTES - 1; //sets the pointer to the correct position: end of incomplete block
		*ptrByte ^= 0x01; //last byte of padding: at the end of the last incomplete block

        //Initializes M[0]
        absorbInput(memMatrixGPU, stateLocal, stateIdxGPU, nBlocksInput);
        reducedSqueezeRow0(memMatrixGPU, stateLocal);
		
		//Initializes M[1]
		reducedDuplexRow1and2(memMatrixGPU, stateLocal, 0, 1);

		//Initializes M[2]
		reducedDuplexRow1and2(memMatrixGPU, stateLocal, 1, 2);

		// end of original bootStrapGPU_P1 code
		
		//============================ Setup, Wandering Phase and Wrap-up =============================//
		//================================ Setup Phase ==================================//
		//==Initializes a (nRows x nCols) memory matrix, it's cells having b bits each)==//
		//============================ Wandering Phase =============================//
		//=====Iteratively overwrites pseudorandom cells of the memory matrix=======//
		//============================ Wrap-up Phase ===============================//
		//========================= Output computation =============================//
		//Absorbs one last block of the memory matrix with the full-round sponge

		int64_t gap = 1;     //Modifier to the step, assuming the values 1 or -1
		uint64_t step = 1;   //Visitation step (used during Setup to dictate the sequence in which rows are read)
		uint64_t window = 2; //Visitation window (used to define which rows can be revisited during Setup)
		uint64_t sqrt = 2;   //Square of window (i.e., square(window)), when a window is a square number;
		                     //otherwise, sqrt = 2*square(window/2)
		uint64_t row0 = 3;   //row0: sequentially written during Setup; randomly picked during Wandering
		uint64_t prev0 = 2;  //prev0: stores the previous value of row0
		uint64_t row1 = 1;   //row1: revisited during Setup, and then read [and written]; randomly picked during Wandering
		uint64_t prev1 = 0;  //prev1: stores the previous value of row1

		for (row0 = 3; row0 < nRows; row0++) {
			//Performs a reduced-round duplexing operation over "M[row1][col] [+] M[prev0][col] [+] M[prev1][col]", filling M[row0] and updating M[row1]
			//M[row0][N_COLS-1-col] = M[prev0][col] XOR rand;
			//M[row1][col] = M[row1][col] XOR rot(rand)                    rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			reducedDuplexRowFilling2OTM_P1(stateLocal, memMatrixGPU, prev0, prev1, row0, row1);

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
		
		wanderingPhaseGPU2_P1(memMatrixGPU, stateLocal, prev0, prev1);
		
		//squeezeGPU(stateLocal, pkeysGPU, kLen);

        outputHash[thread] = ((uint2*)stateLocal)[0];
        outputHash[thread + threads] = ((uint2*)stateLocal)[1];
        outputHash[thread + 2 * threads] = ((uint2*)stateLocal)[2];
        outputHash[thread + 3 * threads] = ((uint2*)stateLocal)[3];
//        ((vectype*)outputHash)[thread] = state[0];

    } //thread
}


__host__
void thebestcoin_cpu_init(int thr_id, uint32_t threads,uint64_t *hash)
{
    cudaMemcpyToSymbol(DMatrix, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
}



__host__ 
void thebestcoin_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t tpb)
{
    dim3 grid((threads + tpb - 1) / tpb);
    dim3 block(tpb);

    thebestcoin_gpu_hash_32 << <grid, block >> > (threads, startNounce, (uint2*)d_outputHash);
}

  
