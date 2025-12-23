#include "galkin_d_sparse_matrix_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

namespace galkin_d_sparse_matrix_mul {

namespace {

constexpr double kEpsDrop = 1e-12;

inline void SplitColumns(int ncols, int size, int rank, int *col_begin, int *col_end) {
  const int base = ncols / size;
  const int rem = ncols % size;
  const int my = base + (rank < rem ? 1 : 0);

  int start = 0;
  if (rank < rem) {
    start = rank * (base + 1);
  } else {
    start = rem * (base + 1) + (rank - rem) * base;
  }

  *col_begin = start;
  *col_end = start + my;
}

// NOLINTNEXTLINE(readability-identifier-naming, readability-function-cognitive-complexity)
inline CCSMatrix MultiplyCCS_ColumnsRange(const CCSMatrix &left, const CCSMatrix &right, int col_begin, int col_end) {
  CCSMatrix local;
  local.nrows = left.nrows;
  local.ncols = col_end - col_begin;
  local.col_ptr.assign(static_cast<std::size_t>(local.ncols) + 1U, 0);

  std::vector<double> acc(static_cast<std::size_t>(local.nrows), 0.0);
  std::vector<int> touched_rows;
  touched_rows.reserve(256);
  std::vector<unsigned char> mark(static_cast<std::size_t>(local.nrows), 0);

  for (int col = col_begin; col < col_end; ++col) {
    const int local_col = col - col_begin;

    for (int row : touched_rows) {
      acc[static_cast<std::size_t>(row)] = 0.0;
      mark[static_cast<std::size_t>(row)] = 0;
    }
    touched_rows.clear();

    for (int pb = right.col_ptr[static_cast<std::size_t>(col)]; pb < right.col_ptr[static_cast<std::size_t>(col + 1)];
         ++pb) {
      const int k = right.row_idx[static_cast<std::size_t>(pb)];
      const double bkj = right.values[static_cast<std::size_t>(pb)];

      for (int pa = left.col_ptr[static_cast<std::size_t>(k)]; pa < left.col_ptr[static_cast<std::size_t>(k + 1)];
           ++pa) {
        const int row = left.row_idx[static_cast<std::size_t>(pa)];
        const double add = left.values[static_cast<std::size_t>(pa)] * bkj;

        if (mark[static_cast<std::size_t>(row)] == 0U) {
          mark[static_cast<std::size_t>(row)] = 1U;
          touched_rows.push_back(row);
        }
        acc[static_cast<std::size_t>(row)] += add;
      }
    }

    std::ranges::sort(touched_rows);
    for (int row : touched_rows) {
      const double value = acc[static_cast<std::size_t>(row)];
      if (std::fabs(value) > kEpsDrop) {
        local.row_idx.push_back(row);
        local.values.push_back(value);
      }
    }

    local.col_ptr[static_cast<std::size_t>(local_col) + 1U] = static_cast<int>(local.values.size());
  }

  return local;
}

inline void BroadcastCCSMatrix(CCSMatrix *matrix, int root, MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  int nrows = 0;
  int ncols = 0;
  int col_ptr_sz = 0;
  int nnz = 0;

  if (rank == root) {
    nrows = matrix->nrows;
    ncols = matrix->ncols;
    col_ptr_sz = static_cast<int>(matrix->col_ptr.size());
    nnz = static_cast<int>(matrix->values.size());
  }

  MPI_Bcast(&nrows, 1, MPI_INT, root, comm);
  MPI_Bcast(&ncols, 1, MPI_INT, root, comm);
  MPI_Bcast(&col_ptr_sz, 1, MPI_INT, root, comm);
  MPI_Bcast(&nnz, 1, MPI_INT, root, comm);

  if (rank != root) {
    matrix->nrows = nrows;
    matrix->ncols = ncols;
    matrix->col_ptr.assign(static_cast<std::size_t>(col_ptr_sz), 0);
    matrix->row_idx.assign(static_cast<std::size_t>(nnz), 0);
    matrix->values.assign(static_cast<std::size_t>(nnz), 0.0);
  }

  if (col_ptr_sz > 0) {
    MPI_Bcast(matrix->col_ptr.data(), col_ptr_sz, MPI_INT, root, comm);
  }
  if (nnz > 0) {
    MPI_Bcast(matrix->row_idx.data(), nnz, MPI_INT, root, comm);
    MPI_Bcast(matrix->values.data(), nnz, MPI_DOUBLE, root, comm);
  }
}

struct GatherLayout {
  std::vector<int> recvcounts;
  std::vector<int> displs;
  int total = 0;
};

inline GatherLayout MakeGatherLayout(const std::vector<int> &counts) {
  GatherLayout layout;
  layout.recvcounts = counts;
  layout.displs.resize(counts.size(), 0);

  int sum = 0;
  for (std::size_t rank_idx = 0; rank_idx < counts.size(); ++rank_idx) {
    layout.displs[rank_idx] = sum;
    sum += counts[rank_idx];
  }
  layout.total = sum;
  return layout;
}

inline CCSMatrix BuildFullFromGathered(const CCSMatrix &left, const CCSMatrix &right, const std::vector<int> &all_cols,
                                       const std::vector<int> &all_nnz, const std::vector<int> &colptr_displs,
                                       const std::vector<int> &gathered_colptr, std::vector<int> &&gathered_row,
                                       std::vector<double> &&gathered_val) {
  CCSMatrix full;
  full.nrows = left.nrows;
  full.ncols = right.ncols;
  full.col_ptr.assign(static_cast<std::size_t>(full.ncols) + 1U, 0);

  full.row_idx = std::move(gathered_row);
  full.values = std::move(gathered_val);

  int global_col = 0;
  int nnz_base = 0;

  for (std::size_t rank_idx = 0; rank_idx < all_cols.size(); ++rank_idx) {
    const int cols_rank = all_cols[rank_idx];
    const int nnz_rank = all_nnz[rank_idx];
    const int cp_off = colptr_displs[rank_idx];

    for (int col_idx = 0; col_idx < cols_rank; ++col_idx) {
      const std::size_t gcol = static_cast<std::size_t>(global_col) + static_cast<std::size_t>(col_idx);
      full.col_ptr[gcol] = nnz_base + gathered_colptr[static_cast<std::size_t>(cp_off + col_idx)];
    }

    global_col += cols_rank;
    nnz_base += nnz_rank;
  }

  full.col_ptr[static_cast<std::size_t>(full.ncols)] = static_cast<int>(full.values.size());
  return full;
}

}  // namespace

GalkinDSparseMatMulMPI::GalkinDSparseMatMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool GalkinDSparseMatMulMPI::ValidationImpl() {
  const auto &in = GetInput();
  return IsValidCCS(in.a) && IsValidCCS(in.b) && IsMultipliable(in.a, in.b);
}

bool GalkinDSparseMatMulMPI::PreProcessingImpl() {
  GetOutput() = OutType{};
  return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool GalkinDSparseMatMulMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (!ValidationImpl()) {
    GetOutput() = OutType{};
    return true;
  }

  const auto &left = GetInput().a;
  const auto &right = GetInput().b;

  int col_begin = 0;
  int col_end = 0;
  SplitColumns(right.ncols, size, rank, &col_begin, &col_end);

  const int my_cols = col_end - col_begin;
  const int my_colptr_len = my_cols + 1;

  CCSMatrix local = MultiplyCCS_ColumnsRange(left, right, col_begin, col_end);
  const int my_nnz = static_cast<int>(local.values.size());

  std::vector<int> all_cols;
  std::vector<int> all_nnz;
  std::vector<int> all_colptr_len;

  if (rank == 0) {
    all_cols.resize(static_cast<std::size_t>(size), 0);
    all_nnz.resize(static_cast<std::size_t>(size), 0);
    all_colptr_len.resize(static_cast<std::size_t>(size), 0);
  }

  MPI_Gather(&my_cols, 1, MPI_INT, rank == 0 ? all_cols.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&my_nnz, 1, MPI_INT, rank == 0 ? all_nnz.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&my_colptr_len, 1, MPI_INT, rank == 0 ? all_colptr_len.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> gathered_colptr;
  std::vector<int> colptr_displs;
  std::vector<int> colptr_recvcounts;

  if (rank == 0) {
    colptr_recvcounts = all_colptr_len;
    const auto layout = MakeGatherLayout(colptr_recvcounts);
    colptr_displs = layout.displs;
    gathered_colptr.assign(static_cast<std::size_t>(layout.total), 0);
  }

  MPI_Gatherv(local.col_ptr.data(), my_colptr_len, MPI_INT, rank == 0 ? gathered_colptr.data() : nullptr,
              rank == 0 ? colptr_recvcounts.data() : nullptr, rank == 0 ? colptr_displs.data() : nullptr, MPI_INT, 0,
              MPI_COMM_WORLD);

  std::vector<int> gathered_row;
  std::vector<double> gathered_val;
  std::vector<int> nnz_displs;
  std::vector<int> nnz_recvcounts;

  if (rank == 0) {
    nnz_recvcounts = all_nnz;
    const auto layout = MakeGatherLayout(nnz_recvcounts);
    nnz_displs = layout.displs;
    gathered_row.assign(static_cast<std::size_t>(layout.total), 0);
    gathered_val.assign(static_cast<std::size_t>(layout.total), 0.0);
  }

  MPI_Gatherv(local.row_idx.data(), my_nnz, MPI_INT, rank == 0 ? gathered_row.data() : nullptr,
              rank == 0 ? nnz_recvcounts.data() : nullptr, rank == 0 ? nnz_displs.data() : nullptr, MPI_INT, 0,
              MPI_COMM_WORLD);

  MPI_Gatherv(local.values.data(), my_nnz, MPI_DOUBLE, rank == 0 ? gathered_val.data() : nullptr,
              rank == 0 ? nnz_recvcounts.data() : nullptr, rank == 0 ? nnz_displs.data() : nullptr, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  CCSMatrix full;
  if (rank == 0) {
    full = BuildFullFromGathered(left, right, all_cols, all_nnz, colptr_displs, gathered_colptr,
                                 std::move(gathered_row), std::move(gathered_val));
  }

  BroadcastCCSMatrix(&full, 0, MPI_COMM_WORLD);
  GetOutput() = std::move(full);
  return true;
}

bool GalkinDSparseMatMulMPI::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_sparse_matrix_mul
