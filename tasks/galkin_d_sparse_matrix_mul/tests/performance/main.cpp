#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "galkin_d_sparse_matrix_mul/common/include/common.hpp"
#include "galkin_d_sparse_matrix_mul/mpi/include/ops_mpi.hpp"
#include "galkin_d_sparse_matrix_mul/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace galkin_d_sparse_matrix_mul {

namespace {

struct XorShift64 {
  uint64_t x;
  explicit XorShift64(uint64_t seed = 88172645463325252ull) : x(seed ? seed : 88172645463325252ull) {}
  uint64_t next_u64() {
    uint64_t z = x;
    z ^= z >> 12;
    z ^= z << 25;
    z ^= z >> 27;
    x = z;
    return z * 2685821657736338717ull;
  }
  double next_double(double lo, double hi) {
    const uint64_t r = next_u64() >> 11;
    const double u = static_cast<double>(r) / static_cast<double>((1ull << 53) - 1ull);
    return lo + (hi - lo) * u;
  }
  int next_int(int lo, int hi) {
    const uint64_t r = next_u64();
    const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
    return lo + static_cast<int>(r % span);
  }
};

CCSMatrix GenerateRandomCCS(int nrows, int ncols, double dens, uint64_t seed) {
  CCSMatrix M;
  M.nrows = nrows;
  M.ncols = ncols;
  M.col_ptr.resize(static_cast<size_t>(ncols + 1), 0);

  XorShift64 rng(seed);

  const int target_per_col = std::max(1, static_cast<int>(std::llround(dens * nrows)));

  std::vector<int> rows;
  rows.reserve(static_cast<size_t>(target_per_col));

  std::vector<unsigned char> used(static_cast<size_t>(nrows), 0);

  for (int j = 0; j < ncols; ++j) {
    rows.clear();

    std::vector<int> touched;
    touched.reserve(static_cast<size_t>(target_per_col));

    while (static_cast<int>(rows.size()) < target_per_col) {
      int r = rng.next_int(0, nrows - 1);
      if (!used[static_cast<size_t>(r)]) {
        used[static_cast<size_t>(r)] = 1;
        touched.push_back(r);
        rows.push_back(r);
      }
    }

    for (int r : touched) {
      used[static_cast<size_t>(r)] = 0;
    }

    std::sort(rows.begin(), rows.end());

    for (int r : rows) {
      M.row_idx.push_back(r);
      double v = rng.next_double(-1.0, 1.0);
      if (std::fabs(v) < 1e-6) {
        v = (v < 0 ? -1.0 : 1.0) * 1e-3;
      }
      M.values.push_back(v);
    }

    M.col_ptr[static_cast<size_t>(j + 1)] = static_cast<int>(M.values.size());
  }

  return M;
}

double ChecksumCCS(const CCSMatrix &C) {
  double s = 0.0;

  for (int j = 0; j < C.ncols; ++j) {
    const int begin = C.col_ptr[j];
    const int end = C.col_ptr[j + 1];
    for (int p = begin; p < end; ++p) {
      const int i = C.row_idx[p];
      const double v = C.values[p];
      s += v * (1.0 + 0.001 * (i + 1)) * (1.0 + 0.0001 * (j + 1));
    }
  }

  return s;
}

}  // namespace

class GalkinDSparseMatMulPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    constexpr int N = 3000;
    constexpr double dens = 0.005;

    input_data_.A = GenerateRandomCCS(N, N, dens, 42ull);
    input_data_.B = GenerateRandomCCS(N, N, dens, 1337ull);
    if (!ppc::util::IsUnderMpirun()) {
      auto task = std::make_shared<GalkinDSparseMatMulSEQ>(input_data_);
      task->Validation();
      task->PreProcessing();
      task->Run();
      task->PostProcessing();
      expected_checksum_ = ChecksumCCS(task->GetOutput());
    } else {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      double cs = 0.0;
      if (rank == 0) {
        auto task = std::make_shared<GalkinDSparseMatMulSEQ>(input_data_);
        task->Validation();
        task->PreProcessing();
        task->Run();
        task->PostProcessing();
        cs = ChecksumCCS(task->GetOutput());
      }
      MPI_Bcast(&cs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      expected_checksum_ = cs;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double cs = ChecksumCCS(output_data);
    const double eps = 1e-7 * (1.0 + std::fabs(expected_checksum_));
    return std::fabs(cs - expected_checksum_) <= eps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  double expected_checksum_ = 0.0;
};

namespace {

TEST_P(GalkinDSparseMatMulPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GalkinDSparseMatMulMPI, GalkinDSparseMatMulSEQ>(
    PPC_SETTINGS_galkin_d_sparse_matrix_mul);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = GalkinDSparseMatMulPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GalkinDSparseMatMulPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace galkin_d_sparse_matrix_mul
