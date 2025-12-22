#include "galkin_d_sparse_matrix_mul/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace galkin_d_sparse_matrix_mul {

namespace {
constexpr double kEpsDrop = 1e-12;
}

GalkinDSparseMatMulSEQ::GalkinDSparseMatMulSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool GalkinDSparseMatMulSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return IsValidCCS(in.A) && IsValidCCS(in.B) && IsMultipliable(in.A, in.B);
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

  const auto &A = GetInput().A;
  const auto &B = GetInput().B;

  CCSMatrix C;
  C.nrows = A.nrows;
  C.ncols = B.ncols;
  C.col_ptr.assign(C.ncols + 1, 0);

  std::vector<double> acc(static_cast<size_t>(C.nrows), 0.0);
  std::vector<int> touched;
  touched.reserve(256);
  std::vector<unsigned char> mark(static_cast<size_t>(C.nrows), 0);

  for (int j = 0; j < B.ncols; ++j) {
    for (int r : touched) {
      acc[static_cast<size_t>(r)] = 0.0;
      mark[static_cast<size_t>(r)] = 0;
    }
    touched.clear();

    for (int pb = B.col_ptr[j]; pb < B.col_ptr[j + 1]; ++pb) {
      const int k = B.row_idx[pb];
      const double bkj = B.values[pb];

      for (int pa = A.col_ptr[k]; pa < A.col_ptr[k + 1]; ++pa) {
        const int i = A.row_idx[pa];
        const double aik = A.values[pa];
        const double add = aik * bkj;

        if (!mark[static_cast<size_t>(i)]) {
          mark[static_cast<size_t>(i)] = 1;
          touched.push_back(i);
        }
        acc[static_cast<size_t>(i)] += add;
      }
    }

    std::sort(touched.begin(), touched.end());
    for (int i : touched) {
      const double v = acc[static_cast<size_t>(i)];
      if (std::fabs(v) > kEpsDrop) {
        C.row_idx.push_back(i);
        C.values.push_back(v);
      }
    }

    C.col_ptr[j + 1] = static_cast<int>(C.values.size());
  }

  GetOutput() = std::move(C);
  return true;
}

bool GalkinDSparseMatMulSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_sparse_matrix_mul
