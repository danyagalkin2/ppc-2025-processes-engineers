#include <gtest/gtest.h>
#include <mpi.h>

#include "galkin_d_ring/common/include/common.hpp"
#include "galkin_d_ring/mpi/include/ops_mpi.hpp"
#include "galkin_d_ring/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace galkin_d_ring {

using InType = galkin_d_ring::InType;
using OutType = galkin_d_ring::OutType;

class GalkinDRingPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int size = 1;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    constexpr int kCount = 5'000'000;
    const int dest = (size > 1) ? size / 2 : 0;

    input_data_ = InType{
        .src = 0,
        .dest = dest,
        .count = kCount,
    };
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{.src = 0, .dest = 0, .count = 1};
};

namespace {

TEST_P(GalkinDRingPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GalkinDRingMPI, GalkinDRingSEQ>(PPC_SETTINGS_galkin_d_ring);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GalkinDRingPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RingPerfSuite, GalkinDRingPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace galkin_d_ring
