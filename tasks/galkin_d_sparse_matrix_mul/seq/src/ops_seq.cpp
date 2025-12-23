#include "galkin_d_sparse_matrix_mul/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "galkin_d_sparse_matrix_mul/common/include/common.hpp"  // include-cleaner

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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
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
      const auto r = static_cast<std::size_t>(row);
      acc[r] = 0.0;
      mark[r] = 0U;
    }
    touched_rows.clear();

    const auto c = static_cast<std::size_t>(col);
    const int pb_begin = right.col_ptr[c];
    const int pb_end = right.col_ptr[c + 1U];

    for (int pb = pb_begin; pb < pb_end; ++pb) {
      const auto p = static_cast<std::size_t>(pb);
      const int k = right.row_idx[p];
      const double bkj = right.values[p];

      const auto kk = static_cast<std::size_t>(k);
      const int pa_begin = left.col_ptr[kk];
      const int pa_end = left.col_ptr[kk + 1U];

      for (int pa = pa_begin; pa < pa_end; ++pa) {
        const auto a = static_cast<std::size_t>(pa);
        const int row = left.row_idx[a];
        const double add = left.values[a] * bkj;

        const auto r = static_cast<std::size_t>(row);
        if (mark[r] == 0U) {
          mark[r] = 1U;
          touched_rows.push_back(row);
        }
        acc[r] += add;
      }
    }

    std::ranges::sort(touched_rows);
    for (int row : touched_rows) {
      const double value = acc[static_cast<std::size_t>(row)];
      if (std::fabs(value) > kEpsDrop) {
        out.row_idx.push_back(row);
        out.values.push_back(value);
      }
    }

    out.col_ptr[c + 1U] = static_cast<int>(out.values.size());
  }

  GetOutput() = std::move(out);
  return true;
}

bool GalkinDSparseMatMulSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_sparse_matrix_mul
