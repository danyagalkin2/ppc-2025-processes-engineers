#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "task/include/task.hpp"

namespace galkin_d_sparse_matrix_mul {

struct CCSMatrix {
  int nrows = 0;
  int ncols = 0;
  std::vector<int> col_ptr;
  std::vector<int> row_idx;
  std::vector<double> values;
};

struct Input {
  CCSMatrix A;
  CCSMatrix B;
};

using InType = Input;
using OutType = CCSMatrix;
using BaseTask = ppc::task::Task<InType, OutType>;

inline bool IsValidCCS(const CCSMatrix &M) {
  if (M.nrows < 0 || M.ncols < 0) {
    return false;
  }
  if (M.col_ptr.size() != static_cast<size_t>(M.ncols + 1)) {
    return false;
  }
  if (M.col_ptr.empty() || M.col_ptr.front() != 0) {
    return false;
  }
  if (M.row_idx.size() != M.values.size()) {
    return false;
  }

  for (int j = 0; j < M.ncols; ++j) {
    if (M.col_ptr[j] > M.col_ptr[j + 1]) {
      return false;
    }
  }

  for (int r : M.row_idx) {
    if (r < 0 || r >= M.nrows) {
      return false;
    }
  }

  return true;
}

inline bool IsMultipliable(const CCSMatrix &A, const CCSMatrix &B) {
  return A.ncols == B.nrows;
}

inline std::vector<double> CCSToDense(const CCSMatrix &M) {
  std::vector<double> dense(M.nrows * M.ncols, 0.0);

  for (int j = 0; j < M.ncols; ++j) {
    for (int p = M.col_ptr[j]; p < M.col_ptr[j + 1]; ++p) {
      int i = M.row_idx[p];
      dense[i * M.ncols + j] += M.values[p];
    }
  }

  return dense;
}

inline CCSMatrix DenseToCCSSorted(const std::vector<double> &dense, int nrows, int ncols, double eps = 1e-12) {
  CCSMatrix M;
  M.nrows = nrows;
  M.ncols = ncols;
  M.col_ptr.resize(ncols + 1, 0);

  for (int j = 0; j < ncols; ++j) {
    for (int i = 0; i < nrows; ++i) {
      double v = dense[i * ncols + j];
      if (std::fabs(v) > eps) {
        M.row_idx.push_back(i);
        M.values.push_back(v);
      }
    }
    M.col_ptr[j + 1] = static_cast<int>(M.values.size());
  }

  return M;
}

inline bool NearlyEqualCCS(const CCSMatrix &A, const CCSMatrix &B, double eps = 1e-9) {
  if (A.nrows != B.nrows || A.ncols != B.ncols) {
    return false;
  }
  if (A.col_ptr != B.col_ptr) {
    return false;
  }
  if (A.row_idx != B.row_idx) {
    return false;
  }
  if (A.values.size() != B.values.size()) {
    return false;
  }

  for (size_t i = 0; i < A.values.size(); ++i) {
    if (std::fabs(A.values[i] - B.values[i]) > eps) {
      return false;
    }
  }

  return true;
}

}  // namespace galkin_d_sparse_matrix_mul
