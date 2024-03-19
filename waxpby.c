#include "waxpby.h"
#include <immintrin.h> 
#include <omp.h>


/**
 * @brief Compute the update of a vector with the sum of two scaled vectors
 * 
 * @param n Number of vector elements
 * @param alpha Scalars applied to x
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */


int waxpby (const int n, const double alpha, const double * const x, const double beta, const double * const y, double * const w) {

  int loopFactor = 4;
  int loopN = (n/loopFactor)*loopFactor;

  __m256d xVec, yVec, wVec;
  __m256d betaVec = _mm256_set1_pd(beta);
  __m256d alphaVec = _mm256_set1_pd(alpha);

  if (alpha==1.0) {

    #pragma omp parallel for private(xVec, yVec, wVec)
    for (int i=0; i<loopN; i+=loopFactor) {

      xVec = _mm256_load_pd(&x[i]);
      yVec = _mm256_load_pd(&y[i]);
      wVec = _mm256_add_pd(xVec, _mm256_mul_pd(yVec, betaVec));
      _mm256_store_pd(&w[i], wVec);

    }
    
    //Handling residual elements
    for (int i = loopN; i<n; i++){
      w[i] = x[i] + beta * y[i];
    }

    
  } else if(beta==1.0) {

    #pragma omp parallel for private(xVec, yVec, wVec)
    for (int i=0; i<loopN; i+=loopFactor) {

      xVec = _mm256_load_pd(&x[i]);
      yVec = _mm256_load_pd(&y[i]);
      wVec = _mm256_add_pd(yVec, _mm256_mul_pd(xVec, alphaVec));
      _mm256_store_pd(&w[i], wVec);

    }

    //Handling residual elements
    for (int i = loopN ; i<n; i++){
      w[i] = alpha * x[i] + y[i];
    }

  } else {

    #pragma omp parallel for private(xVec, yVec, wVec)
    for (int i=0; i<loopN; i+=loopFactor) {
      
      xVec = _mm256_load_pd(&x[i]);
      yVec = _mm256_load_pd(&y[i]);
      wVec = _mm256_add_pd(_mm256_mul_pd(alphaVec, xVec), _mm256_mul_pd(betaVec, yVec));
      _mm256_store_pd(&w[i], wVec);

    }
    
     for (int i = loopN ; i<n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
     }
  }

  return 0;
}
