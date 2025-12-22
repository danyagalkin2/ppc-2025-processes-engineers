#include "galkin_d_sparse_matrix_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace galkin_d_sparse_matrix_mul {

namespace {
constexpr double kEpsDrop = 1e-12;

inline void SplitColumns(int ncols, int size, int rank, int *j_begin, int *j_end) {
  const int base = ncols / size;
  const int rem = ncols % size;
  const int my = base + (rank < rem ? 1 : 0);

  int start = 0;
  if (rank < rem) {
    start = rank * (base + 1);
  } else {
    start = rem * (base + 1) + (rank - rem) * base;
  }

  *j_begin = start;
  *j_end = start + my;
}

inline CCSMatrix MultiplyCCS_ColumnsRange(const CCSMatrix &A, const CCSMatrix &B, int j_begin, int j_end) {
  CCSMatrix C;
  C.nrows = A.nrows;
  C.ncols = j_end - j_begin;
  C.col_ptr.assign(C.ncols + 1, 0);

  std::vector<double> acc(static_cast<size_t>(C.nrows), 0.0);
  std::vector<int> touched;
  touched.reserve(256);
  std::vector<unsigned char> mark(static_cast<size_t>(C.nrows), 0);

  for (int j = j_begin; j < j_end; ++j) {
    const int local_col = j - j_begin;

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
        const double add = A.values[pa] * bkj;

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

    C.col_ptr[local_col + 1] = static_cast<int>(C.values.size());
  }

  return C;
}

inline void BroadcastCCSMatrix(CCSMatrix *M, int root, MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  int nrows = 0, ncols = 0;
  int col_ptr_sz = 0, nnz = 0;

  if (rank == root) {
    nrows = M->nrows;
    ncols = M->ncols;
    col_ptr_sz = static_cast<int>(M->col_ptr.size());
    nnz = static_cast<int>(M->values.size());
  }

  MPI_Bcast(&nrows, 1, MPI_INT, root, comm);
  MPI_Bcast(&ncols, 1, MPI_INT, root, comm);
  MPI_Bcast(&col_ptr_sz, 1, MPI_INT, root, comm);
  MPI_Bcast(&nnz, 1, MPI_INT, root, comm);

  if (rank != root) {
    M->nrows = nrows;
    M->ncols = ncols;
    M->col_ptr.assign(static_cast<size_t>(col_ptr_sz), 0);
    M->row_idx.assign(static_cast<size_t>(nnz), 0);
    M->values.assign(static_cast<size_t>(nnz), 0.0);
  }

  if (col_ptr_sz > 0) {
    MPI_Bcast(M->col_ptr.data(), col_ptr_sz, MPI_INT, root, comm);
  }
  if (nnz > 0) {
    MPI_Bcast(M->row_idx.data(), nnz, MPI_INT, root, comm);
    MPI_Bcast(M->values.data(), nnz, MPI_DOUBLE, root, comm);
  }
}

}  // namespace

GalkinDSparseMatMulMPI::GalkinDSparseMatMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool GalkinDSparseMatMulMPI::ValidationImpl() {
  const auto &in = GetInput();
  return IsValidCCS(in.A) && IsValidCCS(in.B) && IsMultipliable(in.A, in.B);
}

bool GalkinDSparseMatMulMPI::PreProcessingImpl() {
  GetOutput() = OutType{};
  return true;
}

bool GalkinDSparseMatMulMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (!ValidationImpl()) {
    GetOutput() = OutType{};
    return true;
  }

  const auto &A = GetInput().A;
  const auto &B = GetInput().B;

  int j_begin = 0, j_end = 0;
  SplitColumns(B.ncols, size, rank, &j_begin, &j_end);
  const int my_cols = j_end - j_begin;

  CCSMatrix C_local = MultiplyCCS_ColumnsRange(A, B, j_begin, j_end);

  int my_nnz = static_cast<int>(C_local.values.size());
  int my_colptr_len = my_cols + 1;

  std::vector<int> all_cols, all_nnz, all_colptr_len;
  if (rank == 0) {
    all_cols.resize(size, 0);
    all_nnz.resize(size, 0);
    all_colptr_len.resize(size, 0);
  }

  MPI_Gather(&my_cols, 1, MPI_INT, rank == 0 ? all_cols.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&my_nnz, 1, MPI_INT, rank == 0 ? all_nnz.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&my_colptr_len, 1, MPI_INT, rank == 0 ? all_colptr_len.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> colptr_displs, colptr_recvcounts, gathered_colptr;
  if (rank == 0) {
    colptr_recvcounts = all_colptr_len;
    colptr_displs.resize(size, 0);
    int s = 0;
    for (int r = 0; r < size; ++r) {
      colptr_displs[r] = s;
      s += colptr_recvcounts[r];
    }
    gathered_colptr.resize(static_cast<size_t>(s), 0);
  }

  MPI_Gatherv(C_local.col_ptr.data(), my_colptr_len, MPI_INT, rank == 0 ? gathered_colptr.data() : nullptr,
              rank == 0 ? colptr_recvcounts.data() : nullptr, rank == 0 ? colptr_displs.data() : nullptr, MPI_INT, 0,
              MPI_COMM_WORLD);

  std::vector<int> nnz_displs, nnz_recvcounts;
  std::vector<int> gathered_row;
  std::vector<double> gathered_val;

  if (rank == 0) {
    nnz_recvcounts = all_nnz;
    nnz_displs.resize(size, 0);
    int s = 0;
    for (int r = 0; r < size; ++r) {
      nnz_displs[r] = s;
      s += nnz_recvcounts[r];
    }
    gathered_row.resize(static_cast<size_t>(s), 0);
    gathered_val.resize(static_cast<size_t>(s), 0.0);
  }

  MPI_Gatherv(C_local.row_idx.data(), my_nnz, MPI_INT, rank == 0 ? gathered_row.data() : nullptr,
              rank == 0 ? nnz_recvcounts.data() : nullptr, rank == 0 ? nnz_displs.data() : nullptr, MPI_INT, 0,
              MPI_COMM_WORLD);

  MPI_Gatherv(C_local.values.data(), my_nnz, MPI_DOUBLE, rank == 0 ? gathered_val.data() : nullptr,
              rank == 0 ? nnz_recvcounts.data() : nullptr, rank == 0 ? nnz_displs.data() : nullptr, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  CCSMatrix C_full;
  if (rank == 0) {
    C_full.nrows = A.nrows;
    C_full.ncols = B.ncols;
    C_full.col_ptr.assign(static_cast<size_t>(C_full.ncols + 1), 0);

    const int total_nnz = static_cast<int>(gathered_val.size());
    C_full.row_idx = std::move(gathered_row);
    C_full.values = std::move(gathered_val);

    int global_col = 0;
    int nnz_base = 0;

    for (int r = 0; r < size; ++r) {
      const int cols_r = all_cols[r];
      const int nnz_r = all_nnz[r];

      const int cp_off = colptr_displs[r];

      for (int c = 0; c < cols_r; ++c) {
        C_full.col_ptr[static_cast<size_t>(global_col + c)] = nnz_base + gathered_colptr[cp_off + c];
      }

      global_col += cols_r;
      nnz_base += nnz_r;
    }

    C_full.col_ptr[static_cast<size_t>(C_full.ncols)] = total_nnz;
  }

  BroadcastCCSMatrix(&C_full, 0, MPI_COMM_WORLD);
  GetOutput() = std::move(C_full);
  return true;
}

bool GalkinDSparseMatMulMPI::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_sparse_matrix_mul
