#include "ddot.h"
#include <immintrin.h> 
#include <omp.h>

int ddot(const int n, const double *const x, const double *const y, double *const result) {  
  double local_result = 0.0;

  int loopFactor = n > 1000 ? 8 : 4; // 调整为8，因为不能使用512位宽度的指令
  int loopN = (n / loopFactor) * loopFactor;

  #pragma omp parallel for reduction(+:local_result)
  for (int i = 0; i < loopN; i += loopFactor) {
    if (loopFactor == 8) {
      // 对于更大的loopFactor，可以尝试分两次加载和计算
      __m256d xVec1 = _mm256_loadu_pd(&x[i]);
      __m256d yVec1 = _mm256_loadu_pd(&y[i]);
      __m256d product1 = _mm256_mul_pd(xVec1, yVec1);
      __m256d sumVec1 = _mm256_hadd_pd(product1, product1);

      __m256d xVec2 = _mm256_loadu_pd(&x[i + 4]);
      __m256d yVec2 = _mm256_loadu_pd(&y[i + 4]);
      __m256d product2 = _mm256_mul_pd(xVec2, yVec2);
      __m256d sumVec2 = _mm256_hadd_pd(product2, product2);

      double localSums1[4], localSums2[4];
      _mm256_storeu_pd(localSums1, sumVec1);
      _mm256_storeu_pd(localSums2, sumVec2);
      local_result += localSums1[0] + localSums1[2] + localSums2[0] + localSums2[2];
    } else {
      // 原有的AVX2路径保持不变
      __m256d xVec = _mm256_loadu_pd(&x[i]);
      __m256d yVec = _mm256_loadu_pd(&y[i]);
      __m256d product = _mm256_mul_pd(xVec, yVec);
      __m256d sumVec = _mm256_hadd_pd(product, product);
      double localSums[4];
      _mm256_storeu_pd(localSums, sumVec);
      local_result += localSums[0] + localSums[2];
    }
  }

  // 处理剩余元素
  for (int i = loopN; i < n; i++) {
    local_result += x[i] * y[i];
  }

  *result = local_result;

  return 0;
}
