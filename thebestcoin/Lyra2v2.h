/**
 * Header file for the Lyra2 Password Hashing Scheme (PHS).
 * 
 * Author: The Lyra PHC team (http://www.lyra2.net/) -- 2015.
 * 
 * This software is hereby placed in the public domain.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef LYRA2V2_H_
#define LYRA2V2_H_

#include "Spongev2.h"

typedef unsigned char byte ;

#ifndef N_COLS
        #define N_COLS 4                                        //Number of columns in the memory matrix
#endif

#ifndef N_ROWS
		#define N_ROWS 4                                        //Number of rows in the memory matrix
#endif

#ifndef nPARALLEL
        #define nPARALLEL 1                                     //Number of parallel threads
#endif

#define ROW_LEN_INT64 (BLOCK_LEN_INT64 * N_COLS)                //Total length of a row: N_COLS blocks
#define ROW_LEN_BYTES (ROW_LEN_INT64 * 8)                       //Number of bytes per row

int LYRA2V2(void *K, unsigned int kLen, const void *pwd, unsigned int pwdlen, const void *salt, unsigned int saltlen, unsigned int timeCost, unsigned int nRows, unsigned int nCols);

#endif /* LYRA2V2_H_ */
