#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>

#include "galkin_d_trapezoid_method/common/include/common.hpp"
#include "galkin_d_trapezoid_method/mpi/include/ops_mpi.hpp"
#include "galkin_d_trapezoid_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

constexpr double kPi = 3.14159265358979323846;

namespace galkin_d_trapezoid_method {

class GalkinDTrapezoidFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int case_id = std::get<0>(params);
    switch (case_id) {
      case 0: {
        input_data_ = {0.0, 1.0, 1000, static_cast<int>(FunctionId::Linear)};
        break;
      }
      case 1: {
        input_data_ = {0.0, 2.0, 2000, static_cast<int>(FunctionId::Quadratic)};
        break;
      }
      case 2: {
        input_data_ = {0.0, kPi, 3000, static_cast<int>(FunctionId::Sin)};
        break;
      }
      case 3: {
        input_data_ = {-1.0, 1.0, 1500, static_cast<int>(FunctionId::Linear)};
        break;
      }
      default: {
        input_data_ = {0.0, 1.0, 1000, static_cast<int>(FunctionId::Linear)};
        break;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double exact = GetExactIntegral(input_data_);
    const double eps = 1e-4;

    return std::fabs(output_data - exact) < eps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{0.0, 1.0, 10, static_cast<int>(FunctionId::Linear)};
};

namespace {

TEST_P(GalkinDTrapezoidFuncTests, ComputesIntegralWithReasonableAccuracy) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kFunctionalParams = {std::make_tuple(0, "linear_0_1"),
                                                   std::make_tuple(1, "quadratic_0_2"), std::make_tuple(2, "sin_0_pi"),
                                                   std::make_tuple(3, "linear_m1_1")};

const auto kTaskMatrix = std::tuple_cat(ppc::util::AddFuncTask<GalkinDTrapezoidMethodMPI, InType>(
                                            kFunctionalParams, PPC_SETTINGS_galkin_d_trapezoid_method),
                                        ppc::util::AddFuncTask<GalkinDTrapezoidMethodSEQ, InType>(
                                            kFunctionalParams, PPC_SETTINGS_galkin_d_trapezoid_method));

const auto kParameterizedValues = ppc::util::ExpandToValues(kTaskMatrix);

const auto kFunctionalTestName = GalkinDTrapezoidFuncTests::PrintFuncTestName<GalkinDTrapezoidFuncTests>;

INSTANTIATE_TEST_SUITE_P(TrapezoidIntegralSuite, GalkinDTrapezoidFuncTests, kParameterizedValues, kFunctionalTestName);

template <typename TaskType>
void ExpectFullPipelineSuccess(const InType &in, double eps = 1e-4) {
  auto task = std::make_shared<TaskType>(in);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());

  const double exact = GetExactIntegral(in);
  ASSERT_NEAR(task->GetOutput(), exact, eps);
}

TEST(GalkinDTrapezoidStandalone, SeqPipelineStandardCases) {
  const std::array<InType, 3> kInputs = {InType{0.0, 1.0, 1000, static_cast<int>(FunctionId::Linear)},
                                         InType{0.0, 2.0, 2000, static_cast<int>(FunctionId::Quadratic)},
                                         InType{0.0, kPi, 4000, static_cast<int>(FunctionId::Sin)}};

  for (const auto &in : kInputs) {
    ExpectFullPipelineSuccess<GalkinDTrapezoidMethodSEQ>(in);
  }
}

TEST(GalkinDTrapezoidStandalone, MpiPipelineStandardCases) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  const std::array<InType, 3> kInputs = {InType{0.0, 1.0, 1000, static_cast<int>(FunctionId::Linear)},
                                         InType{0.0, 2.0, 2000, static_cast<int>(FunctionId::Quadratic)},
                                         InType{0.0, kPi, 4000, static_cast<int>(FunctionId::Sin)}};

  for (const auto &in : kInputs) {
    ExpectFullPipelineSuccess<GalkinDTrapezoidMethodMPI>(in);
  }
}

TEST(GalkinDTrapezoidValidation, RejectsNonPositiveNSeq) {
  InType in{0.0, 1.0, 0, static_cast<int>(FunctionId::Linear)};
  GalkinDTrapezoidMethodSEQ task(in);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(GalkinDTrapezoidValidation, RejectsNonPositiveNMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType in{0.0, 1.0, 0, static_cast<int>(FunctionId::Linear)};
  GalkinDTrapezoidMethodMPI task(in);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(GalkinDTrapezoidValidation, RejectsInvalidIntervalSeq) {
  InType in{1.0, 0.0, 100, static_cast<int>(FunctionId::Linear)};
  GalkinDTrapezoidMethodSEQ task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDTrapezoidValidation, RejectsInvalidIntervalMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType in{1.0, 0.0, 100, static_cast<int>(FunctionId::Linear)};
  GalkinDTrapezoidMethodMPI task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDTrapezoidValidation, AcceptsValidInputSeq) {
  InType in{0.0, 1.0, 1000, static_cast<int>(FunctionId::Linear)};
  GalkinDTrapezoidMethodSEQ task(in);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  const double exact = GetExactIntegral(in);
  EXPECT_NEAR(task.GetOutput(), exact, 1e-4);
}

TEST(GalkinDTrapezoidValidation, AcceptsValidInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType in{0.0, 1.0, 1000, static_cast<int>(FunctionId::Linear)};
  GalkinDTrapezoidMethodMPI task(in);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  const double exact = GetExactIntegral(in);
  EXPECT_NEAR(task.GetOutput(), exact, 1e-4);
}

template <typename TaskType>
void RunTaskTwice(TaskType &task, const InType &first, const InType &second, double eps = 1e-4) {
  task.GetInput() = first;
  task.GetOutput() = 0.0;
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_NEAR(task.GetOutput(), GetExactIntegral(first), eps);

  task.GetInput() = second;
  task.GetOutput() = 0.0;
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_NEAR(task.GetOutput(), GetExactIntegral(second), eps);
}

TEST(GalkinDTrapezoidPipeline, SeqTaskCanBeReusedAcrossRuns) {
  InType first{0.0, 1.0, 1000, static_cast<int>(FunctionId::Linear)};
  InType second{0.0, kPi, 3000, static_cast<int>(FunctionId::Sin)};

  GalkinDTrapezoidMethodSEQ task(first);
  RunTaskTwice(task, first, second);
}

TEST(GalkinDTrapezoidPipeline, MpiTaskCanBeReusedAcrossRuns) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  InType first{0.0, 1.0, 1000, static_cast<int>(FunctionId::Linear)};
  InType second{0.0, kPi, 3000, static_cast<int>(FunctionId::Sin)};

  GalkinDTrapezoidMethodMPI task(first);
  RunTaskTwice(task, first, second);
}

}  // namespace

}  // namespace galkin_d_trapezoid_method
