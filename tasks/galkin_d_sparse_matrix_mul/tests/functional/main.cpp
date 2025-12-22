#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
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

std::vector<double> DenseMatMul(const std::vector<double> &A, int n, int k, const std::vector<double> &B, int m) {
  std::vector<double> C(static_cast<size_t>(n) * static_cast<size_t>(m), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int t = 0; t < k; ++t) {
      const double a = A[static_cast<size_t>(i) * k + t];
      if (std::fabs(a) < 1e-15) {
        continue;
      }
      for (int j = 0; j < m; ++j) {
        C[static_cast<size_t>(i) * m + j] += a * B[static_cast<size_t>(t) * m + j];
      }
    }
  }
  return C;
}

CCSMatrix MakeIdentityCCS(int n) {
  CCSMatrix I;
  I.nrows = n;
  I.ncols = n;
  I.col_ptr.resize(static_cast<size_t>(n + 1), 0);
  I.row_idx.reserve(static_cast<size_t>(n));
  I.values.reserve(static_cast<size_t>(n));
  for (int j = 0; j < n; ++j) {
    I.row_idx.push_back(j);
    I.values.push_back(1.0);
    I.col_ptr[static_cast<size_t>(j + 1)] = j + 1;
  }
  return I;
}

CCSMatrix MakeDiagonalCCS(const std::vector<double> &diag) {
  const int n = static_cast<int>(diag.size());
  CCSMatrix D;
  D.nrows = n;
  D.ncols = n;
  D.col_ptr.resize(static_cast<size_t>(n + 1), 0);
  for (int j = 0; j < n; ++j) {
    if (std::fabs(diag[static_cast<size_t>(j)]) > 1e-15) {
      D.row_idx.push_back(j);
      D.values.push_back(diag[static_cast<size_t>(j)]);
    }
    D.col_ptr[static_cast<size_t>(j + 1)] = static_cast<int>(D.values.size());
  }
  return D;
}

CCSMatrix MakeSmallManualA_3x2() {
  CCSMatrix A;
  A.nrows = 3;
  A.ncols = 2;
  A.col_ptr = {0, 2, 4};
  A.row_idx = {0, 1, 1, 2};
  A.values = {1.0, 2.0, 3.0, 4.0};
  return A;
}

CCSMatrix MakeSmallManualB_2x3() {
  CCSMatrix B;
  B.nrows = 2;
  B.ncols = 3;
  B.col_ptr = {0, 2, 3, 4};
  B.row_idx = {0, 1, 1, 0};
  B.values = {5.0, 7.0, 8.0, 6.0};
  return B;
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
        input_data_.A = MakeIdentityCCS(n);
        input_data_.B = MakeIdentityCCS(n);

        auto Ad = CCSToDense(input_data_.A);
        auto Bd = CCSToDense(input_data_.B);
        auto Cd = DenseMatMul(Ad, n, n, Bd, n);
        expected_ = DenseToCCSSorted(Cd, n, n);
        break;
      }
      case 1: {
        const int n = 5;
        std::vector<double> diag = {2.0, -1.0, 0.0, 3.5, 4.0};
        input_data_.A = MakeDiagonalCCS(diag);
        input_data_.B = MakeIdentityCCS(n);

        auto Ad = CCSToDense(input_data_.A);
        auto Bd = CCSToDense(input_data_.B);
        auto Cd = DenseMatMul(Ad, n, n, Bd, n);
        expected_ = DenseToCCSSorted(Cd, n, n);
        break;
      }
      case 2: {
        input_data_.A = MakeSmallManualA_3x2();
        input_data_.B = MakeSmallManualB_2x3();

        auto Ad = CCSToDense(input_data_.A);
        auto Bd = CCSToDense(input_data_.B);
        auto Cd = DenseMatMul(Ad, 3, 2, Bd, 3);
        expected_ = DenseToCCSSorted(Cd, 3, 3);
        break;
      }
      case 3: {
        CCSMatrix A;
        A.nrows = 2;
        A.ncols = 3;
        A.col_ptr = {0, 1, 2, 4};
        A.row_idx = {0, 1, 0, 1};
        A.values = {1.0, 3.0, 2.0, 4.0};

        CCSMatrix B;
        B.nrows = 3;
        B.ncols = 2;
        B.col_ptr = {0, 2, 4};
        B.row_idx = {0, 1, 1, 2};
        B.values = {5.0, 6.0, 7.0, 8.0};

        input_data_.A = A;
        input_data_.B = B;

        auto Ad = CCSToDense(A);
        auto Bd = CCSToDense(B);
        auto Cd = DenseMatMul(Ad, 2, 3, Bd, 2);
        expected_ = DenseToCCSSorted(Cd, 2, 2);
        break;
      }
      default: {
        const int n = 3;
        input_data_.A = MakeIdentityCCS(n);
        input_data_.B = MakeIdentityCCS(n);

        auto Ad = CCSToDense(input_data_.A);
        auto Bd = CCSToDense(input_data_.B);
        auto Cd = DenseMatMul(Ad, n, n, Bd, n);
        expected_ = DenseToCCSSorted(Cd, n, n);
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
  in.A = MakeSmallManualA_3x2();
  in.B = MakeSmallManualB_2x3();

  auto Ad = CCSToDense(in.A);
  auto Bd = CCSToDense(in.B);
  auto Cd = DenseMatMul(Ad, 3, 2, Bd, 3);
  OutType expected = DenseToCCSSorted(Cd, 3, 3);

  ExpectFullPipelineSuccess<GalkinDSparseMatMulSEQ>(in, expected);
}

TEST(GalkinDSparseMatMulStandalone, MpiPipelineStandardCase) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  InType in;
  in.A = MakeSmallManualA_3x2();
  in.B = MakeSmallManualB_2x3();

  auto Ad = CCSToDense(in.A);
  auto Bd = CCSToDense(in.B);
  auto Cd = DenseMatMul(Ad, 3, 2, Bd, 3);
  OutType expected = DenseToCCSSorted(Cd, 3, 3);

  ExpectFullPipelineSuccess<GalkinDSparseMatMulMPI>(in, expected);
}

TEST(GalkinDSparseMatMulValidation, RejectsIncompatibleDimsSeq) {
  InType in;
  in.A = MakeIdentityCCS(3);
  in.B = MakeIdentityCCS(4);
  GalkinDSparseMatMulSEQ task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDSparseMatMulValidation, RejectsIncompatibleDimsMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType in;
  in.A = MakeIdentityCCS(3);
  in.B = MakeIdentityCCS(4);
  GalkinDSparseMatMulMPI task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDSparseMatMulValidation, RejectsInvalidCCSSeq) {
  InType in;
  in.A = MakeIdentityCCS(3);
  in.B = MakeIdentityCCS(3);
  in.A.col_ptr = {1, 0, 0, 0};
  GalkinDSparseMatMulSEQ task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDSparseMatMulValidation, RejectsInvalidCCSMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType in;
  in.A = MakeIdentityCCS(3);
  in.B = MakeIdentityCCS(3);
  in.B.col_ptr = {0, 2, 1, 3};
  GalkinDSparseMatMulMPI task(in);
  EXPECT_FALSE(task.Validation());
}

template <typename TaskType>
void RunTaskTwice(TaskType &task, const InType &first, const OutType &exp1, const InType &second, const OutType &exp2) {
  task.GetInput() = first;
  task.GetOutput() = OutType{};
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_TRUE(NearlyEqualCCS(task.GetOutput(), exp1, 1e-9));

  task.GetInput() = second;
  task.GetOutput() = OutType{};
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_TRUE(NearlyEqualCCS(task.GetOutput(), exp2, 1e-9));
}

TEST(GalkinDSparseMatMulPipeline, SeqTaskCanBeReusedAcrossRuns) {
  InType first;
  first.A = MakeIdentityCCS(4);
  first.B = MakeIdentityCCS(4);

  auto Ad1 = CCSToDense(first.A);
  auto Bd1 = CCSToDense(first.B);
  auto Cd1 = DenseMatMul(Ad1, 4, 4, Bd1, 4);
  OutType exp1 = DenseToCCSSorted(Cd1, 4, 4);

  InType second;
  second.A = MakeSmallManualA_3x2();
  second.B = MakeSmallManualB_2x3();

  auto Ad2 = CCSToDense(second.A);
  auto Bd2 = CCSToDense(second.B);
  auto Cd2 = DenseMatMul(Ad2, 3, 2, Bd2, 3);
  OutType exp2 = DenseToCCSSorted(Cd2, 3, 3);

  GalkinDSparseMatMulSEQ task(first);
  RunTaskTwice(task, first, exp1, second, exp2);
}

TEST(GalkinDSparseMatMulPipeline, MpiTaskCanBeReusedAcrossRuns) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  InType first;
  first.A = MakeIdentityCCS(4);
  first.B = MakeIdentityCCS(4);

  auto Ad1 = CCSToDense(first.A);
  auto Bd1 = CCSToDense(first.B);
  auto Cd1 = DenseMatMul(Ad1, 4, 4, Bd1, 4);
  OutType exp1 = DenseToCCSSorted(Cd1, 4, 4);

  InType second;
  second.A = MakeSmallManualA_3x2();
  second.B = MakeSmallManualB_2x3();

  auto Ad2 = CCSToDense(second.A);
  auto Bd2 = CCSToDense(second.B);
  auto Cd2 = DenseMatMul(Ad2, 3, 2, Bd2, 3);
  OutType exp2 = DenseToCCSSorted(Cd2, 3, 3);

  GalkinDSparseMatMulMPI task(first);
  RunTaskTwice(task, first, exp1, second, exp2);
}

}  // namespace

}  // namespace galkin_d_sparse_matrix_mul
