#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
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
  CCSMatrix a;
  CCSMatrix b;
};

using InType = Input;
using OutType = CCSMatrix;
using BaseTask = ppc::task::Task<InType, OutType>;

inline bool IsValidCCS(const CCSMatrix &matrix) {
  if (matrix.nrows < 0 || matrix.ncols < 0) {
    return false;
  }

  const auto expected_col_ptr = static_cast<std::size_t>(matrix.ncols) + 1U;
  if (matrix.col_ptr.size() != expected_col_ptr) {
    return false;
  }
  if (matrix.col_ptr.empty() || matrix.col_ptr.front() != 0) {
    return false;
  }
  if (matrix.row_idx.size() != matrix.values.size()) {
    return false;
  }

  for (int col = 0; col < matrix.ncols; ++col) {
    const auto c = static_cast<std::size_t>(col);
    if (matrix.col_ptr[c] > matrix.col_ptr[c + 1U]) {
      return false;
    }
  }

  return std::ranges::all_of(matrix.row_idx, [&](int row) { return row >= 0 && row < matrix.nrows; });
}

inline bool IsMultipliable(const CCSMatrix &left, const CCSMatrix &right) {
  return left.ncols == right.nrows;
}

inline std::vector<double> CCSToDense(const CCSMatrix &matrix) {
  const auto total = static_cast<std::size_t>(matrix.nrows) * static_cast<std::size_t>(matrix.ncols);
  std::vector<double> dense(total, 0.0);

  for (int col = 0; col < matrix.ncols; ++col) {
    const auto c = static_cast<std::size_t>(col);
    const int begin = matrix.col_ptr[c];
    const int end = matrix.col_ptr[c + 1U];

    for (int pos = begin; pos < end; ++pos) {
      const auto p = static_cast<std::size_t>(pos);
      const int row = matrix.row_idx[p];
      const auto idx = (static_cast<std::size_t>(row) * static_cast<std::size_t>(matrix.ncols)) + c;
      dense[idx] += matrix.values[p];
    }
  }

  return dense;
}

inline CCSMatrix DenseToCCSSorted(const std::vector<double> &dense, int nrows, int ncols, double eps = 1e-12) {
  CCSMatrix matrix;
  matrix.nrows = nrows;
  matrix.ncols = ncols;
  matrix.col_ptr.resize(static_cast<std::size_t>(ncols) + 1U, 0);

  for (int col = 0; col < ncols; ++col) {
    const auto c = static_cast<std::size_t>(col);
    for (int row = 0; row < nrows; ++row) {
      const auto idx = (static_cast<std::size_t>(row) * static_cast<std::size_t>(ncols)) + c;
      const double value = dense[idx];
      if (std::fabs(value) > eps) {
        matrix.row_idx.push_back(row);
        matrix.values.push_back(value);
      }
    }
    matrix.col_ptr[c + 1U] = static_cast<int>(matrix.values.size());
  }

  return matrix;
}

inline bool NearlyEqualCCS(const CCSMatrix &left, const CCSMatrix &right, double eps = 1e-9) {
  if (left.nrows != right.nrows || left.ncols != right.ncols) {
    return false;
  }
  if (left.col_ptr != right.col_ptr) {
    return false;
  }
  if (left.row_idx != right.row_idx) {
    return false;
  }
  if (left.values.size() != right.values.size()) {
    return false;
  }

  for (std::size_t idx = 0; idx < left.values.size(); ++idx) {
    if (std::fabs(left.values[idx] - right.values[idx]) > eps) {
      return false;
    }
  }

  return true;
}

}  // namespace galkin_d_sparse_matrix_mul
