#include "ddot.h"
#include <immintrin.h> 
#include <omp.h>

/**
 * @brief Compute the dot product of two vectors
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */
int ddot (const int n, const double * const x, const double * const y, double * const result) {  

  double local_result = 0.0;

  int loopFactor = 4;
  int loopN = (n/loopFactor) * loopFactor;

  #pragma omp parallel for reduction(+:local_result)
  for (int i=0; i<loopN; i+=loopFactor) {

    __m256d xVec = _mm256_load_pd(&x[i]);
    __m256d yVec = _mm256_load_pd(&y[i]);
    __m256d product = _mm256_mul_pd(xVec, yVec);
    __m256d sumVec = _mm256_hadd_pd(product, product);
    double localSums[4];

    _mm256_storeu_pd(localSums, sumVec);
    local_result += localSums[0] + localSums[2];
  }

  for(int i = loopN; i < n; i++){
    local_result += x[i]*y[i];
}

*result = local_result;

return 0;
}
