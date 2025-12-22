#include "galkin_d_sparse_matrix_mul/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace galkin_d_sparse_matrix_mul {

namespace {
constexpr double kEpsDrop = 1e-12;
}  // namespace

GalkinDSparseMatMulSEQ::GalkinDSparseMatMulSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool GalkinDSparseMatMulSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return IsValidCCS(in.a) && IsValidCCS(in.b) && IsMultipliable(in.a, in.b);
}

bool GalkinDSparseMatMulSEQ::PreProcessingImpl() {
  GetOutput() = OutType{};
  return true;
}

bool GalkinDSparseMatMulSEQ::RunImpl() {
  if (!ValidationImpl()) {
    GetOutput() = OutType{};
    return true;
  }

  const auto &left = GetInput().a;
  const auto &right = GetInput().b;

  CCSMatrix out;
  out.nrows = left.nrows;
  out.ncols = right.ncols;
  out.col_ptr.assign(static_cast<std::size_t>(out.ncols) + 1U, 0);

  std::vector<double> acc(static_cast<std::size_t>(out.nrows), 0.0);
  std::vector<int> touched_rows;
  touched_rows.reserve(256);
  std::vector<unsigned char> mark(static_cast<std::size_t>(out.nrows), 0);

  for (int col = 0; col < right.ncols; ++col) {
    for (int row : touched_rows) {
      acc[static_cast<std::size_t>(row)] = 0.0;
      mark[static_cast<std::size_t>(row)] = 0;
    }
    touched_rows.clear();

    for (int pb = right.col_ptr[col]; pb < right.col_ptr[col + 1]; ++pb) {
      const int k = right.row_idx[pb];
      const double bkj = right.values[pb];

      for (int pa = left.col_ptr[k]; pa < left.col_ptr[k + 1]; ++pa) {
        const int row = left.row_idx[pa];
        const double add = left.values[pa] * bkj;

        if (mark[static_cast<std::size_t>(row)] == 0U) {
          mark[static_cast<std::size_t>(row)] = 1U;
          touched_rows.push_back(row);
        }
        acc[static_cast<std::size_t>(row)] += add;
      }
    }

    std::sort(touched_rows.begin(), touched_rows.end());
    for (int row : touched_rows) {
      const double value = acc[static_cast<std::size_t>(row)];
      if (std::fabs(value) > kEpsDrop) {
        out.row_idx.push_back(row);
        out.values.push_back(value);
      }
    }

    out.col_ptr[static_cast<std::size_t>(col) + 1U] = static_cast<int>(out.values.size());
  }

  GetOutput() = std::move(out);
  return true;
}

bool GalkinDSparseMatMulSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_sparse_matrix_mul
