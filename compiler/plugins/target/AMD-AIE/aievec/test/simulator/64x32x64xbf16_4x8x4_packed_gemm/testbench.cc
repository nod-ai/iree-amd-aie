#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define MAT_A_SIZE 2048
#define MAT_B_SIZE 2048
#define MAT_C_SIZE 4096
#define N_SIZE 64
#define M_SIZE 64
#define K_SIZE 32

bfloat16 mat_a_data[MAT_A_SIZE];
bfloat16 mat_b_data[MAT_B_SIZE];
float mat_c_data[MAT_C_SIZE];
float ref_c_data[MAT_C_SIZE];

#define INPUT_A_FILE "matrix_a_test.txt"
#define INPUT_B_FILE "matrix_b_test.txt"
#define OUTPUT_C_FILE "matrix_c_test.txt"

#ifndef __chess__
int chess_cycle_count() { return 0; }
#endif

extern void gemm_64x32x64_bf16_packed_4x8x4(bfloat16 *restrict mat_a_data,
                                            bfloat16 *restrict mat_b_data,
                                            float *restrict mat_c_data);

int main() {
  int i = 0, j = 0, k = 0;

  // Read in matrix_a to local memory
  int index = 0;
  for (i = 0; i < N_SIZE; i++) {
    for (k = 0; k < K_SIZE; k++) {
      int32_t ival = *reinterpret_cast<int32_t *>(&i);
      int16_t bfval = (ival & 0xFFFF0000) >> 16;
      mat_a_data[index++] = *reinterpret_cast<bfloat16 *>(&bfval);
    }
  }

  // Read in matrix_b to local memory
  index = 0;
  for (k = 0; k < K_SIZE; k++) {
    for (j = 0; j < M_SIZE; j++) {
      int32_t ival = *reinterpret_cast<int32_t *>(&i);
      int16_t bfval = (ival & 0xFFFF0000) >> 16;
      mat_b_data[index++] = *reinterpret_cast<bfloat16 *>(&bfval);
    }
  }

  // Initialize matrix_c to local memory
  index = 0;
  for (i = 0; i < N_SIZE; i++) {
    for (j = 0; j < M_SIZE; j++) {
      mat_c_data[index++] = 0.f;
    }
  }

  // Compute matrix multiplication
  // reference(mat_a_data, mat_b_data, mat_c_data);
  auto cyclesBegin = chess_cycle_count();
  gemm_64x32x64_bf16_packed_4x8x4(mat_a_data, mat_b_data, mat_c_data);
  auto cyclesEnd = chess_cycle_count();

  return 0;
}
