#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "galkin_d_sparse_matrix_mul/common/include/common.hpp"
#include "galkin_d_sparse_matrix_mul/mpi/include/ops_mpi.hpp"
#include "galkin_d_sparse_matrix_mul/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace galkin_d_sparse_matrix_mul {

namespace {

struct XorShift64 {
  static constexpr std::uint64_t kDefaultSeed = 88172645463325252ULL;

  std::uint64_t x;

  explicit XorShift64(std::uint64_t seed = kDefaultSeed) : x(seed) {
    if (x == 0ULL) {
      x = kDefaultSeed;
    }
  }

  std::uint64_t NextU64() {
    std::uint64_t z = x;
    z ^= (z >> 12);
    z ^= (z << 25);
    z ^= (z >> 27);
    x = z;
    return z * 2685821657736338717ULL;
  }

  double NextDouble(double lo, double hi) {
    const std::uint64_t r = NextU64() >> 11;
    const double u = static_cast<double>(r) / static_cast<double>((1ULL << 53) - 1ULL);
    return lo + ((hi - lo) * u);
  }

  int NextInt(int lo, int hi) {
    const std::uint64_t r = NextU64();
    const std::uint64_t span = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo) + 1ULL;
    return lo + static_cast<int>(r % span);
  }
};

CCSMatrix GenerateRandomCCS(int nrows, int ncols, double dens, std::uint64_t seed) {
  CCSMatrix matrix;
  matrix.nrows = nrows;
  matrix.ncols = ncols;
  matrix.col_ptr.resize(static_cast<std::size_t>(ncols) + 1U, 0);

  XorShift64 rng(seed);

  const int target_per_col = std::max(1, static_cast<int>(std::llround(dens * nrows)));

  std::vector<int> rows;
  rows.reserve(static_cast<std::size_t>(target_per_col));

  std::vector<unsigned char> used(static_cast<std::size_t>(nrows), 0);

  for (int col = 0; col < ncols; ++col) {
    rows.clear();

    std::vector<int> touched_rows;
    touched_rows.reserve(static_cast<std::size_t>(target_per_col));

    while (rows.size() < static_cast<std::size_t>(target_per_col)) {
      const int row = rng.NextInt(0, nrows - 1);
      if (used[static_cast<std::size_t>(row)] == 0U) {
        used[static_cast<std::size_t>(row)] = 1U;
        touched_rows.push_back(row);
        rows.push_back(row);
      }
    }

    for (int row : touched_rows) {
      used[static_cast<std::size_t>(row)] = 0U;
    }

    std::ranges::sort(rows);

    for (int row : rows) {
      matrix.row_idx.push_back(row);
      double value = rng.NextDouble(-1.0, 1.0);
      if (std::fabs(value) < 1e-6) {
        value = (value < 0.0 ? -1.0 : 1.0) * 1e-3;
      }
      matrix.values.push_back(value);
    }

    matrix.col_ptr[static_cast<std::size_t>(col) + 1U] = static_cast<int>(matrix.values.size());
  }

  return matrix;
}

double ChecksumCCS(const CCSMatrix &matrix) {
  double sum = 0.0;

  for (int col = 0; col < matrix.ncols; ++col) {
    const auto c = static_cast<std::size_t>(col);
    const int begin = matrix.col_ptr[c];
    const int end = matrix.col_ptr[c + 1U];

    for (int pos = begin; pos < end; ++pos) {
      const auto p = static_cast<std::size_t>(pos);
      const int row = matrix.row_idx[p];
      const double value = matrix.values[p];
      sum += value * (1.0 + 0.001 * (row + 1)) * (1.0 + 0.0001 * (col + 1));
    }
  }

  return sum;
}

}  // namespace

class GalkinDSparseMatMulPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    constexpr int kN = 3000;
    constexpr double kDens = 0.005;

    input_data_.a = GenerateRandomCCS(kN, kN, kDens, 42ULL);
    input_data_.b = GenerateRandomCCS(kN, kN, kDens, 1337ULL);

    if (!ppc::util::IsUnderMpirun()) {
      auto task = std::make_shared<GalkinDSparseMatMulSEQ>(input_data_);
      task->Validation();
      task->PreProcessing();
      task->Run();
      task->PostProcessing();
      expected_checksum_ = ChecksumCCS(task->GetOutput());
      return;
    }

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
