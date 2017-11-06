#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>


#include "../../cuda_helper.h"
#include "../../cuda_vector.h"
#include "../cuda_vad.h"
#include "../utils.h"


#define vectype uint28
__device__ vectype  *DMatrix;
static _ALIGN(64) uint64_t * gpu_out  = NULL;
static _ALIGN(64) uint64_t * gpu_matrix = NULL;
