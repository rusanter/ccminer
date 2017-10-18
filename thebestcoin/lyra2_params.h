#ifndef LYRA2_PARAMS_H_
#define LYRA2_PARAMS_H_

// Change these params of Lyra2 algoryhtm:

#define LYRA2_ROWS  4	// Number of rows in the memory matrix
#define LYRA2_COLS  4	// Number of columns in the memory matrix
#define LYRA2_TCOST 1

// Do NOT CHANGE the following params.

#define nPARALLEL   1	// Number of parallel threads in one hashing function
#define LYRA2_RHO   1	// Number of reduced rounds performed

// Block length required so Blake2's Initialization Vector (IV) is not overwritten (THIS SHOULD NOT BE MODIFIED)
#define BLOCK_LEN_BLAKE2_SAFE_INT64 8                                   // 512 bits (=64 bytes, =8 uint64_t)
#define BLOCK_LEN_BLAKE2_SAFE_BYTES (BLOCK_LEN_BLAKE2_SAFE_INT64 * 8)   // same as above, in bytes

// Default block lenght: 768 bits
#define BLOCK_LEN_INT64 12								// Block length: 768 bits (=96 bytes, =12 uint64_t)
#define BLOCK_LEN_BYTES (BLOCK_LEN_INT64 * 8)			// Block length, in bytes

#define ROW_LEN_INT64 (BLOCK_LEN_INT64 * LYRA2_COLS)	// Total length of a row
#define ROW_LEN_BYTES (ROW_LEN_INT64 * 8)				// Number of bytes per row

#endif
