#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "galkin_d_sparse_matrix_mul/common/include/common.hpp"
#include "galkin_d_sparse_matrix_mul/mpi/include/ops_mpi.hpp"
#include "galkin_d_sparse_matrix_mul/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace galkin_d_sparse_matrix_mul {

using TestType = std::tuple<int, std::string>;

namespace {

std::vector<double> DenseMatMul(const std::vector<double> &left, int nrows, int shared_dim,
                                const std::vector<double> &right, int ncols) {
  const std::size_t total = static_cast<std::size_t>(nrows) * static_cast<std::size_t>(ncols);
  std::vector<double> out(total, 0.0);

  for (int row = 0; row < nrows; ++row) {
    for (int kk = 0; kk < shared_dim; ++kk) {
      const std::size_t left_idx =
          (static_cast<std::size_t>(row) * static_cast<std::size_t>(shared_dim)) + static_cast<std::size_t>(kk);
      const double a_val = left[left_idx];
      if (std::fabs(a_val) < 1e-15) {
        continue;
      }
      for (int col = 0; col < ncols; ++col) {
        const std::size_t out_idx =
            (static_cast<std::size_t>(row) * static_cast<std::size_t>(ncols)) + static_cast<std::size_t>(col);
        const std::size_t right_idx =
            (static_cast<std::size_t>(kk) * static_cast<std::size_t>(ncols)) + static_cast<std::size_t>(col);
        out[out_idx] += a_val * right[right_idx];
      }
    }
  }
  return out;
}

CCSMatrix MakeIdentityCCS(int n) {
  CCSMatrix identity;
  identity.nrows = n;
  identity.ncols = n;
  identity.col_ptr.resize(static_cast<std::size_t>(n) + 1U, 0);
  identity.row_idx.reserve(static_cast<std::size_t>(n));
  identity.values.reserve(static_cast<std::size_t>(n));

  for (int col = 0; col < n; ++col) {
    identity.row_idx.push_back(col);
    identity.values.push_back(1.0);
    identity.col_ptr[static_cast<std::size_t>(col) + 1U] = col + 1;
  }
  return identity;
}

CCSMatrix MakeDiagonalCCS(const std::vector<double> &diag) {
  const int n = static_cast<int>(diag.size());
  CCSMatrix diagonal;
  diagonal.nrows = n;
  diagonal.ncols = n;
  diagonal.col_ptr.resize(static_cast<std::size_t>(n) + 1U, 0);

  for (int col = 0; col < n; ++col) {
    const double val = diag[static_cast<std::size_t>(col)];
    if (std::fabs(val) > 1e-15) {
      diagonal.row_idx.push_back(col);
      diagonal.values.push_back(val);
    }
    diagonal.col_ptr[static_cast<std::size_t>(col) + 1U] = static_cast<int>(diagonal.values.size());
  }
  return diagonal;
}

CCSMatrix MakeSmallManualA3x2() {
  CCSMatrix a;
  a.nrows = 3;
  a.ncols = 2;
  a.col_ptr = {0, 2, 4};
  a.row_idx = {0, 1, 1, 2};
  a.values = {1.0, 2.0, 3.0, 4.0};
  return a;
}

CCSMatrix MakeSmallManualB2x3() {
  CCSMatrix b;
  b.nrows = 2;
  b.ncols = 3;
  b.col_ptr = {0, 2, 3, 4};
  b.row_idx = {0, 1, 1, 0};
  b.values = {5.0, 7.0, 8.0, 6.0};
  return b;
}

}  // namespace

class GalkinDSparseMatMulFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    const auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int case_id = std::get<0>(params);

    switch (case_id) {
      case 0: {
        const int n = 4;
        input_data_.a = MakeIdentityCCS(n);
        input_data_.b = MakeIdentityCCS(n);

        const auto dense_a = CCSToDense(input_data_.a);
        const auto dense_b = CCSToDense(input_data_.b);
        const auto dense_c = DenseMatMul(dense_a, n, n, dense_b, n);
        expected_ = DenseToCCSSorted(dense_c, n, n);
        break;
      }
      case 1: {
        const int n = 5;
        const std::vector<double> diag = {2.0, -1.0, 0.0, 3.5, 4.0};
        input_data_.a = MakeDiagonalCCS(diag);
        input_data_.b = MakeIdentityCCS(n);

        const auto dense_a = CCSToDense(input_data_.a);
        const auto dense_b = CCSToDense(input_data_.b);
        const auto dense_c = DenseMatMul(dense_a, n, n, dense_b, n);
        expected_ = DenseToCCSSorted(dense_c, n, n);
        break;
      }
      case 2: {
        input_data_.a = MakeSmallManualA3x2();
        input_data_.b = MakeSmallManualB2x3();

        const auto dense_a = CCSToDense(input_data_.a);
        const auto dense_b = CCSToDense(input_data_.b);
        const auto dense_c = DenseMatMul(dense_a, 3, 2, dense_b, 3);
        expected_ = DenseToCCSSorted(dense_c, 3, 3);
        break;
      }
      case 3: {
        CCSMatrix a;
        a.nrows = 2;
        a.ncols = 3;
        a.col_ptr = {0, 1, 2, 4};
        a.row_idx = {0, 1, 0, 1};
        a.values = {1.0, 3.0, 2.0, 4.0};

        CCSMatrix b;
        b.nrows = 3;
        b.ncols = 2;
        b.col_ptr = {0, 2, 4};
        b.row_idx = {0, 1, 1, 2};
        b.values = {5.0, 6.0, 7.0, 8.0};

        input_data_.a = a;
        input_data_.b = b;

        const auto dense_a = CCSToDense(a);
        const auto dense_b = CCSToDense(b);
        const auto dense_c = DenseMatMul(dense_a, 2, 3, dense_b, 2);
        expected_ = DenseToCCSSorted(dense_c, 2, 2);
        break;
      }
      default: {
        const int n = 3;
        input_data_.a = MakeIdentityCCS(n);
        input_data_.b = MakeIdentityCCS(n);

        const auto dense_a = CCSToDense(input_data_.a);
        const auto dense_b = CCSToDense(input_data_.b);
        const auto dense_c = DenseMatMul(dense_a, n, n, dense_b, n);
        expected_ = DenseToCCSSorted(dense_c, n, n);
        break;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return NearlyEqualCCS(output_data, expected_, 1e-9);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_{};
};

namespace {

TEST_P(GalkinDSparseMatMulFuncTests, ComputesCorrectProduct) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kFunctionalParams = {
    std::make_tuple(0, "I4_I4"),
    std::make_tuple(1, "Diag5_I5"),
    std::make_tuple(2, "Manual_3x2_2x3"),
    std::make_tuple(3, "Rect_2x3_3x2"),
};

const auto kTaskMatrix = std::tuple_cat(
    ppc::util::AddFuncTask<GalkinDSparseMatMulMPI, InType>(kFunctionalParams, PPC_SETTINGS_galkin_d_sparse_matrix_mul),
    ppc::util::AddFuncTask<GalkinDSparseMatMulSEQ, InType>(kFunctionalParams, PPC_SETTINGS_galkin_d_sparse_matrix_mul));

const auto kParameterizedValues = ppc::util::ExpandToValues(kTaskMatrix);
const auto kFunctionalTestName = GalkinDSparseMatMulFuncTests::PrintFuncTestName<GalkinDSparseMatMulFuncTests>;

INSTANTIATE_TEST_SUITE_P(SparseMatMulFunctionalSuite, GalkinDSparseMatMulFuncTests, kParameterizedValues,
                         kFunctionalTestName);

template <typename TaskType>
void ExpectFullPipelineSuccess(const InType &in, const OutType &expected) {
  auto task = std::make_shared<TaskType>(in);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_TRUE(NearlyEqualCCS(task->GetOutput(), expected, 1e-9));
}

TEST(GalkinDSparseMatMulStandalone, SeqPipelineStandardCase) {
  InType in;
  in.a = MakeSmallManualA3x2();
  in.b = MakeSmallManualB2x3();

  const auto dense_a = CCSToDense(in.a);
  const auto dense_b = CCSToDense(in.b);
  const auto dense_c = DenseMatMul(dense_a, 3, 2, dense_b, 3);
  const OutType expected = DenseToCCSSorted(dense_c, 3, 3);

  ExpectFullPipelineSuccess<GalkinDSparseMatMulSEQ>(in, expected);
}

TEST(GalkinDSparseMatMulStandalone, MpiPipelineStandardCase) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  InType in;
  in.a = MakeSmallManualA3x2();
  in.b = MakeSmallManualB2x3();

  const auto dense_a = CCSToDense(in.a);
  const auto dense_b = CCSToDense(in.b);
  const auto dense_c = DenseMatMul(dense_a, 3, 2, dense_b, 3);
  const OutType expected = DenseToCCSSorted(dense_c, 3, 3);

  ExpectFullPipelineSuccess<GalkinDSparseMatMulMPI>(in, expected);
}

TEST(GalkinDSparseMatMulValidation, RejectsIncompatibleDimsSeq) {
  InType in;
  in.a = MakeIdentityCCS(3);
  in.b = MakeIdentityCCS(4);
  GalkinDSparseMatMulSEQ task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDSparseMatMulValidation, RejectsIncompatibleDimsMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType in;
  in.a = MakeIdentityCCS(3);
  in.b = MakeIdentityCCS(4);
  GalkinDSparseMatMulMPI task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDSparseMatMulValidation, RejectsInvalidCCSSeq) {
  InType in;
  in.a = MakeIdentityCCS(3);
  in.b = MakeIdentityCCS(3);
  in.b.col_ptr = {1, 0, 0, 0};
  GalkinDSparseMatMulSEQ task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDSparseMatMulValidation, RejectsInvalidCCSMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType in;
  in.a = MakeIdentityCCS(3);
  in.b = MakeIdentityCCS(3);
  in.b.col_ptr = {0, 2, 1, 3};
  GalkinDSparseMatMulMPI task(in);
  EXPECT_FALSE(task.Validation());
}

template <typename TaskType>
void RunTaskTwice(TaskType &task, const InType &first, const OutType &expected_first, const InType &second,
                  const OutType &expected_second) {
  task.GetInput() = first;
  task.GetOutput() = OutType{};
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_TRUE(NearlyEqualCCS(task.GetOutput(), expected_first, 1e-9));

  task.GetInput() = second;
  task.GetOutput() = OutType{};
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_TRUE(NearlyEqualCCS(task.GetOutput(), expected_second, 1e-9));
}

TEST(GalkinDSparseMatMulPipeline, SeqTaskCanBeReusedAcrossRuns) {
  InType first;
  first.a = MakeIdentityCCS(4);
  first.b = MakeIdentityCCS(4);

  const auto dense_a1 = CCSToDense(first.a);
  const auto dense_b1 = CCSToDense(first.b);
  const auto dense_c1 = DenseMatMul(dense_a1, 4, 4, dense_b1, 4);
  const OutType expected_first = DenseToCCSSorted(dense_c1, 4, 4);

  InType second;
  second.a = MakeSmallManualA3x2();
  second.b = MakeSmallManualB2x3();

  const auto dense_a2 = CCSToDense(second.a);
  const auto dense_b2 = CCSToDense(second.b);
  const auto dense_c2 = DenseMatMul(dense_a2, 3, 2, dense_b2, 3);
  const OutType expected_second = DenseToCCSSorted(dense_c2, 3, 3);

  GalkinDSparseMatMulSEQ task(first);
  RunTaskTwice(task, first, expected_first, second, expected_second);
}

TEST(GalkinDSparseMatMulPipeline, MpiTaskCanBeReusedAcrossRuns) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  InType first;
  first.a = MakeIdentityCCS(4);
  first.b = MakeIdentityCCS(4);

  const auto dense_a1 = CCSToDense(first.a);
  const auto dense_b1 = CCSToDense(first.b);
  const auto dense_c1 = DenseMatMul(dense_a1, 4, 4, dense_b1, 4);
  const OutType expected_first = DenseToCCSSorted(dense_c1, 4, 4);

  InType second;
  second.a = MakeSmallManualA3x2();
  second.b = MakeSmallManualB2x3();

  const auto dense_a2 = CCSToDense(second.a);
  const auto dense_b2 = CCSToDense(second.b);
  const auto dense_c2 = DenseMatMul(dense_a2, 3, 2, dense_b2, 3);
  const OutType expected_second = DenseToCCSSorted(dense_c2, 3, 3);

  GalkinDSparseMatMulMPI task(first);
  RunTaskTwice(task, first, expected_first, second, expected_second);
}

}  // namespace

}  // namespace galkin_d_sparse_matrix_mul
