#include "galkin_d_sparse_matrix_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "galkin_d_sparse_matrix_mul/common/include/common.hpp"

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
    start = (rem * (base + 1)) + ((rank - rem) * base);
  }

  *col_begin = start;
  *col_end = start + my;
}

inline void ResetTouched(std::vector<double> &acc, std::vector<unsigned char> &mark, std::vector<int> &touched_rows) {
  for (int row : touched_rows) {
    const auto r = static_cast<std::size_t>(row);
    acc[r] = 0.0;
    mark[r] = 0U;
  }
  touched_rows.clear();
}

inline void AddToAccumulator(int row, double add, std::vector<double> &acc, std::vector<unsigned char> &mark,
                             std::vector<int> &touched_rows) {
  const auto r = static_cast<std::size_t>(row);
  if (mark[r] == 0U) {
    mark[r] = 1U;
    touched_rows.push_back(row);
  }
  acc[r] += add;
}

inline CCSMatrix MultiplyCcsColumnsRange(const CCSMatrix &left, const CCSMatrix &right, int col_begin, int col_end) {
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
    ResetTouched(acc, mark, touched_rows);

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
        AddToAccumulator(row, left.values[a] * bkj, acc, mark, touched_rows);
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

    const auto cp = static_cast<std::size_t>(cp_off);
    for (int col_idx = 0; col_idx < cols_rank; ++col_idx) {
      const auto gcol = static_cast<std::size_t>(global_col) + static_cast<std::size_t>(col_idx);
      full.col_ptr[gcol] = nnz_base + gathered_colptr[cp + static_cast<std::size_t>(col_idx)];
    }

    global_col += cols_rank;
    nnz_base += nnz_rank;
  }

  full.col_ptr[static_cast<std::size_t>(full.ncols)] = static_cast<int>(full.values.size());
  return full;
}

struct RootMetaRecv {
  int *cols = nullptr;
  int *nnz = nullptr;
  int *colptr_len = nullptr;
};

inline RootMetaRecv MakeRootMetaRecv(int rank, std::vector<int> *all_cols, std::vector<int> *all_nnz,
                                     std::vector<int> *all_colptr_len) {
  RootMetaRecv r;
  if (rank == 0) {
    r.cols = all_cols->data();
    r.nnz = all_nnz->data();
    r.colptr_len = all_colptr_len->data();
  }
  return r;
}

inline void GatherMetaToRoot(int rank, int size, int my_cols, int my_nnz, int my_colptr_len, std::vector<int> *all_cols,
                             std::vector<int> *all_nnz, std::vector<int> *all_colptr_len) {
  if (rank == 0) {
    all_cols->assign(static_cast<std::size_t>(size), 0);
    all_nnz->assign(static_cast<std::size_t>(size), 0);
    all_colptr_len->assign(static_cast<std::size_t>(size), 0);
  }

  const auto recv = MakeRootMetaRecv(rank, all_cols, all_nnz, all_colptr_len);

  MPI_Gather(&my_cols, 1, MPI_INT, recv.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&my_nnz, 1, MPI_INT, recv.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&my_colptr_len, 1, MPI_INT, recv.colptr_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

struct RootGathervRecvI32 {
  int *data = nullptr;
  int *counts = nullptr;
  int *displs = nullptr;
};

struct RootGathervRecvF64 {
  double *data = nullptr;
  int *counts = nullptr;
  int *displs = nullptr;
};

inline RootGathervRecvI32 MakeRootRecvI32(int rank, std::vector<int> *buf, std::vector<int> *counts,
                                          std::vector<int> *displs) {
  RootGathervRecvI32 r;
  if (rank == 0) {
    r.data = buf->data();
    r.counts = counts->data();
    r.displs = displs->data();
  }
  return r;
}

inline RootGathervRecvF64 MakeRootRecvF64(int rank, std::vector<double> *buf, std::vector<int> *counts,
                                          std::vector<int> *displs) {
  RootGathervRecvF64 r;
  if (rank == 0) {
    r.data = buf->data();
    r.counts = counts->data();
    r.displs = displs->data();
  }
  return r;
}

inline void GatherColPtrToRoot(int rank, const CCSMatrix &local, int my_colptr_len,
                               const std::vector<int> &all_colptr_len, std::vector<int> *gathered_colptr,
                               std::vector<int> *colptr_displs, std::vector<int> *colptr_recvcounts) {
  if (rank == 0) {
    *colptr_recvcounts = all_colptr_len;
    const auto layout = MakeGatherLayout(*colptr_recvcounts);
    *colptr_displs = layout.displs;
    gathered_colptr->assign(static_cast<std::size_t>(layout.total), 0);
  }

  const auto recv = MakeRootRecvI32(rank, gathered_colptr, colptr_recvcounts, colptr_displs);

  MPI_Gatherv(local.col_ptr.data(), my_colptr_len, MPI_INT, recv.data, recv.counts, recv.displs, MPI_INT, 0,
              MPI_COMM_WORLD);
}

inline void GatherNnzToRoot(int rank, const CCSMatrix &local, int my_nnz, const std::vector<int> &all_nnz,
                            std::vector<int> *gathered_row, std::vector<double> *gathered_val,
                            std::vector<int> *nnz_displs, std::vector<int> *nnz_recvcounts) {
  if (rank == 0) {
    *nnz_recvcounts = all_nnz;
    const auto layout = MakeGatherLayout(*nnz_recvcounts);
    *nnz_displs = layout.displs;
    gathered_row->assign(static_cast<std::size_t>(layout.total), 0);
    gathered_val->assign(static_cast<std::size_t>(layout.total), 0.0);
  }

  const auto recv_i32 = MakeRootRecvI32(rank, gathered_row, nnz_recvcounts, nnz_displs);
  const auto recv_f64 = MakeRootRecvF64(rank, gathered_val, nnz_recvcounts, nnz_displs);

  MPI_Gatherv(local.row_idx.data(), my_nnz, MPI_INT, recv_i32.data, recv_i32.counts, recv_i32.displs, MPI_INT, 0,
              MPI_COMM_WORLD);

  MPI_Gatherv(local.values.data(), my_nnz, MPI_DOUBLE, recv_f64.data, recv_f64.counts, recv_f64.displs, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
}

inline CCSMatrix BuildAndBroadcastResult(int rank, const CCSMatrix &left, const CCSMatrix &right,
                                         const std::vector<int> &all_cols, const std::vector<int> &all_nnz,
                                         const std::vector<int> &colptr_displs, const std::vector<int> &gathered_colptr,
                                         std::vector<int> &&gathered_row, std::vector<double> &&gathered_val) {
  CCSMatrix full;
  if (rank == 0) {
    full = BuildFullFromGathered(left, right, all_cols, all_nnz, colptr_displs, gathered_colptr,
                                 std::move(gathered_row), std::move(gathered_val));
  }
  BroadcastCCSMatrix(&full, 0, MPI_COMM_WORLD);
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

bool GalkinDSparseMatMulMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int ok_local = ValidationImpl() ? 1 : 0;
  int ok_all = 0;
  MPI_Allreduce(&ok_local, &ok_all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

  if (ok_all == 0) {
    GetOutput() = OutType{};
    return true;
  }

  CCSMatrix left = GetInput().a;
  CCSMatrix right = GetInput().b;

  BroadcastCCSMatrix(&left, 0, MPI_COMM_WORLD);
  BroadcastCCSMatrix(&right, 0, MPI_COMM_WORLD);

  int col_begin = 0;
  int col_end = 0;
  SplitColumns(right.ncols, size, rank, &col_begin, &col_end);

  const int my_cols = col_end - col_begin;
  const int my_colptr_len = my_cols + 1;

  CCSMatrix local = MultiplyCcsColumnsRange(left, right, col_begin, col_end);
  const int my_nnz = static_cast<int>(local.values.size());

  std::vector<int> all_cols;
  std::vector<int> all_nnz;
  std::vector<int> all_colptr_len;

  GatherMetaToRoot(rank, size, my_cols, my_nnz, my_colptr_len, &all_cols, &all_nnz, &all_colptr_len);

  std::vector<int> gathered_colptr;
  std::vector<int> colptr_displs;
  std::vector<int> colptr_recvcounts;

  GatherColPtrToRoot(rank, local, my_colptr_len, all_colptr_len, &gathered_colptr, &colptr_displs, &colptr_recvcounts);

  std::vector<int> gathered_row;
  std::vector<double> gathered_val;
  std::vector<int> nnz_displs;
  std::vector<int> nnz_recvcounts;

  GatherNnzToRoot(rank, local, my_nnz, all_nnz, &gathered_row, &gathered_val, &nnz_displs, &nnz_recvcounts);

  CCSMatrix full = BuildAndBroadcastResult(rank, left, right, all_cols, all_nnz, colptr_displs, gathered_colptr,
                                           std::move(gathered_row), std::move(gathered_val));

  GetOutput() = std::move(full);
  return true;
}

bool GalkinDSparseMatMulMPI::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_sparse_matrix_mul
