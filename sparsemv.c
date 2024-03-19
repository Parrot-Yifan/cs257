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


  #pragma omp parallel for
  for (int i=0; i< nrow; i++) {
      double sum = 0.0;
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
      const int cur_nnz = (const int) A->nnz_in_row[i];

      int j = 0;
      for (; j <= cur_nnz - 8; j += 8) {
        sum += cur_vals[j]*x[cur_inds[j]] + cur_vals[j + 1]*x[cur_inds[j + 1]] + cur_vals[j + 2]*x[cur_inds[j + 2]] + cur_vals[j + 3]*x[cur_inds[j + 3]] + cur_vals[j + 4]*x[cur_inds[j + 4]] + cur_vals[j + 5]*x[cur_inds[j + 5]] + cur_vals[j + 6]*x[cur_inds[j + 6]] + cur_vals[j + 7]*x[cur_inds[j + 7]];
        // sum += cur_vals[j + 1]*x[cur_inds[j + 1]];
        // sum += cur_vals[j + 2]*x[cur_inds[j + 2]];
        // sum += cur_vals[j + 3]*x[cur_inds[j + 3]];
      }

      for (; j < cur_nnz; j++) {
        sum += cur_vals[j] * x[cur_inds[j]];
      }

      y[i] = sum;
    }
  return 0;
}