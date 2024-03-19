#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>

#include "sparsemv.h"

#include <immintrin.h> 
#include <omp.h>

/**
 * @brief Compute matrix vector product (y = A*x)
 * 
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */
int sparsemv(struct mesh *A, const double * const x, double * const y) {

  const int nrow = (const int) A->local_nrow;
  double sum = 0.0;


  #pragma omp parallel for reduction(+:sum)
  for (int i=0; i< nrow; i++) {
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
      const int cur_nnz = (const int) A->nnz_in_row[i];

      int j = 0;
      for (; j <= cur_nnz - 4; j += 4) {
        sum += cur_vals[j]*x[cur_inds[j]];
        sum += cur_vals[j + 1]*x[cur_inds[j + 1]];
        sum += cur_vals[j + 2]*x[cur_inds[j + 2]];
        sum += cur_vals[j + 3]*x[cur_inds[j + 3]];
      }

      for (; j < cur_nnz; j++) {
        sum += cur_vals[j] * x[cur_inds[j]];
      }

      y[i] = sum;
    }
  return 0;
}