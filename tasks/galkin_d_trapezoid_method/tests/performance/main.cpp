#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

#include "galkin_d_trapezoid_method/common/include/common.hpp"
#include "galkin_d_trapezoid_method/mpi/include/ops_mpi.hpp"
#include "galkin_d_trapezoid_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

constexpr double kPi = 3.14159265358979323846;

namespace galkin_d_trapezoid_method {

class GalkinDTrapezoidPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    constexpr int kCount = 2'000'000;

    input_data_ = InType{0.0, kPi, kCount, static_cast<int>(FunctionId::Sin)};
  }
  bool CheckTestOutputData(OutType &output_data) final {
    const double exact = GetExactIntegral(input_data_);
    constexpr double kEps = 1e-4;

    return std::fabs(output_data - exact) < kEps;
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{0.0, 1.0, 10, static_cast<int>(FunctionId::Linear)};
};

namespace {

TEST_P(GalkinDTrapezoidPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GalkinDTrapezoidMethodMPI, GalkinDTrapezoidMethodSEQ>(
    PPC_SETTINGS_galkin_d_trapezoid_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GalkinDTrapezoidPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GalkinDTrapezoidPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace galkin_d_trapezoid_method
