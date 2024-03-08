
#include <inttypes.h>
#include <stddef.h>

#define MLP_M 128
#define MLP_K 128
#define MLP_N 128

typedef struct {
  const float* lhs;
  const float* lhs_aligned;
  size_t lhs_offset;
  size_t lhs_size0;
  size_t lhs_size1;
  size_t lhs_stride0;
  size_t lhs_stride1;
  const float* rhs;
  const float* rhs_aligned;
  size_t rhs_offset;
  size_t rhs_size0;
  size_t rhs_size1;
  size_t rhs_stride0;
  size_t rhs_stride1;
  float* result;
  float* result_aligned;
  size_t result_offset;
  size_t result_size0;
  size_t result_size1;
  size_t result_stride0;
  size_t result_stride1;
  int32_t M;
  int32_t N;
  int32_t K;
} params_t;

#ifdef __cplusplus
extern "C" {
#endif

int setupNPUAccelerator();
int aie_matmul(const params_t *params);

#ifdef __cplusplus
}
#endif
