#include "galkin_d_trapezoid_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>

#include "galkin_d_trapezoid_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace galkin_d_trapezoid_method {

GalkinDTrapezoidMethodMPI::GalkinDTrapezoidMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool GalkinDTrapezoidMethodMPI::ValidationImpl() {
  const auto &in = GetInput();
  return (in.n > 0) && (in.b > in.a);
}

bool GalkinDTrapezoidMethodMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool GalkinDTrapezoidMethodMPI::RunImpl() {
  const auto &in = GetInput();
  double a = in.a;
  double b = in.b;
  int n = in.n;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int base = n / size;
  int rem = n % size;

  int local_n = base + (rank < rem ? 1 : 0);

  int start_i = rank * base + std::min(rank, rem);
  int end_i = start_i + local_n;
  double h = (b - a) / static_cast<double>(n);

  double local_sum = 0.0;
  for (int i = start_i; i < end_i; ++i) {
    double x_left = a + i * h;
    double x_right = a + (i + 1) * h;

    double f_left = Function(x_left, in.func_id);
    double f_right = Function(x_right, in.func_id);

    local_sum += (f_left + f_right) * 0.5 * h;
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_sum;

  return true;
}

bool GalkinDTrapezoidMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_trapezoid_method
