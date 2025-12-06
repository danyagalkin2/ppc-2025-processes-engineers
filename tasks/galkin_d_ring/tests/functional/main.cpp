#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <string>
#include <tuple>

#include "galkin_d_ring/common/include/common.hpp"
#include "galkin_d_ring/mpi/include/ops_mpi.hpp"
#include "galkin_d_ring/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace galkin_d_ring {

using ::testing::Test;
using ::testing::TestParamInfo;

using InType = galkin_d_ring::InType;
using OutType = galkin_d_ring::OutType;
using TestType = std::tuple<int, std::string>;

class GalkinDRingFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    const auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int case_id = std::get<0>(params);

    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 1) {
      world_size = 1;
    }

    switch (case_id) {
      case 0: {
        input_data_ = InType{
            .src = 0,
            .dest = world_size - 1,
            .count = 16,
        };
        break;
      }
      case 1: {
        const int dest = (world_size > 1) ? 1 : 0;
        input_data_ = InType{
            .src = 0,
            .dest = dest,
            .count = 8,
        };
        break;
      }
      case 2: {
        const int src = world_size / 2;
        input_data_ = InType{
            .src = src,
            .dest = 0,
            .count = 32,
        };
        break;
      }
      case 3: {
        input_data_ = InType{
            .src = 0,
            .dest = 0,
            .count = 10,
        };
        break;
      }
      default: {
        input_data_ = InType{
            .src = 0,
            .dest = world_size - 1,
            .count = 4,
        };
        break;
      }
    }
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

const std::array<TestType, 4> kFunctionalParams = {
    std::make_tuple(0, "0_to_last"),
    std::make_tuple(1, "0_to_neighbor"),
    std::make_tuple(2, "mid_to_0"),
    std::make_tuple(3, "src_equals_dest"),
};

const auto kTaskMatrix =
    std::tuple_cat(ppc::util::AddFuncTask<GalkinDRingMPI, InType>(kFunctionalParams, PPC_SETTINGS_galkin_d_ring),
                   ppc::util::AddFuncTask<GalkinDRingSEQ, InType>(kFunctionalParams, PPC_SETTINGS_galkin_d_ring));

const auto kParameterizedValues = ppc::util::ExpandToValues(kTaskMatrix);

const auto kFunctionalTestName = GalkinDRingFuncTests::PrintFuncTestName<GalkinDRingFuncTests>;

INSTANTIATE_TEST_SUITE_P(RingFunctionalSuite, GalkinDRingFuncTests, kParameterizedValues, kFunctionalTestName);

TEST_P(GalkinDRingFuncTests, TransfersCorrectly) {
  ExecuteTest(GetParam());
}

template <typename TaskType>
void ExpectFullPipelineSuccess(const InType &in) {
  auto task = std::make_shared<TaskType>(in);

  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());

  EXPECT_EQ(task->GetOutput(), 1);
}

TEST(GalkinDRingValidation, RejectsNonPositiveCountSeq) {
  InType in{.src = 0, .dest = 1, .count = 0};
  GalkinDRingSEQ task(in);

  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(GalkinDRingValidation, RejectsNonPositiveCountMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  InType in{.src = 0, .dest = 1, .count = 0};
  GalkinDRingMPI task(in);

  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(GalkinDRingValidation, RejectsInvalidSrcSeq) {
  InType in{.src = -1, .dest = 0, .count = 10};
  GalkinDRingSEQ task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDRingValidation, RejectsInvalidSrcMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  InType in{.src = -1, .dest = 0, .count = 10};
  GalkinDRingMPI task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDRingValidation, RejectsInvalidDestSeq) {
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  InType in{.src = 0, .dest = world_size, .count = 10};
  GalkinDRingSEQ task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDRingValidation, RejectsInvalidDestMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  InType in{.src = 0, .dest = world_size, .count = 10};
  GalkinDRingMPI task(in);
  EXPECT_FALSE(task.Validation());
}

TEST(GalkinDRingValidation, AcceptsValidInputSeq) {
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int dest = (world_size > 1) ? 1 : 0;

  InType in{.src = 0, .dest = dest, .count = 32};
  GalkinDRingSEQ task(in);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

TEST(GalkinDRingValidation, AcceptsValidInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int dest = (world_size > 1) ? 1 : 0;

  InType in{.src = 0, .dest = dest, .count = 32};
  GalkinDRingMPI task(in);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

template <typename TaskType>
void RunTaskTwice(TaskType &task, const InType &first, const InType &second) {
  task.GetInput() = first;
  task.GetOutput() = 0;
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_EQ(task.GetOutput(), 1);

  task.GetInput() = second;
  task.GetOutput() = 0;
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_EQ(task.GetOutput(), 1);
}

TEST(GalkinDRingPipeline, SeqTaskCanBeReusedAcrossRuns) {
  InType first{.src = 0, .dest = 0, .count = 10};
  InType second{.src = 0, .dest = 0, .count = 32};

  GalkinDRingSEQ task(first);
  RunTaskTwice(task, first, second);
}

TEST(GalkinDRingPipeline, MpiTaskCanBeReusedAcrossRuns) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  InType first{.src = 0, .dest = 0, .count = 10};
  InType second{.src = 0, .dest = 0, .count = 32};

  GalkinDRingMPI task(first);
  RunTaskTwice(task, first, second);
}

}  // namespace

}  // namespace galkin_d_ring
