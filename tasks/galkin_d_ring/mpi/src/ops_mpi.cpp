#include "galkin_d_ring/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "galkin_d_ring/common/include/common.hpp"

namespace galkin_d_ring {

namespace {

struct CommGuard {
  MPI_Comm *comm = nullptr;

  explicit CommGuard(MPI_Comm *c) : comm(c) {}

  ~CommGuard() {
    if (comm != nullptr && *comm != MPI_COMM_NULL) {
      MPI_Comm_free(comm);
      *comm = MPI_COMM_NULL;
    }
  }
};

bool ValidateParams(const InType &in, int size) {
  return (in.count > 0) && (in.src >= 0) && (in.src < size) && (in.dest >= 0) && (in.dest < size);
}

std::vector<int> InitBuffer(int rank, int src, int count) {
  std::vector<int> buffer(static_cast<std::size_t>(count), 0);
  if (rank == src) {
    for (std::size_t i = 0; i < buffer.size(); ++i) {
      buffer[i] = static_cast<int>(i) + 1;
    }
  }
  return buffer;
}

void RingTransfer(MPI_Comm comm, int rank, int size, int src, int dest, std::vector<int> &buffer) {
  const int count = static_cast<int>(buffer.size());
  const int steps = (dest - src + size) % size;

  for (int step = 1; step <= steps; ++step) {
    const int sender = (src + step - 1) % size;
    const int receiver = (src + step) % size;

    if (rank == sender) {
      MPI_Send(buffer.data(), count, MPI_INT, receiver, 0, comm);
    } else if (rank == receiver) {
      MPI_Recv(buffer.data(), count, MPI_INT, sender, 0, comm, MPI_STATUS_IGNORE);
    }
  }
}

int CheckAndReduce(MPI_Comm comm, int rank, int dest, const std::vector<int> &buffer) {
  int local_ok = 1;

  if (rank == dest) {
    for (std::size_t i = 0; i < buffer.size(); ++i) {
      if (buffer[i] != static_cast<int>(i) + 1) {
        local_ok = 0;
        break;
      }
    }
  }

  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, comm);
  return global_ok;
}

}  // namespace

GalkinDRingMPI::GalkinDRingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GalkinDRingMPI::ValidationImpl() {
  const auto &in = GetInput();

  if (in.count <= 0) {
    return false;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  size = std::max(size, 1);

  if (in.src < 0 || in.src >= size) {
    return false;
  }
  if (in.dest < 0 || in.dest >= size) {
    return false;
  }

  return true;
}

bool GalkinDRingMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool GalkinDRingMPI::RunImpl() {
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  CommGuard guard(&comm);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const auto in = GetInput();

  if (!ValidateParams(in, size)) {
    GetOutput() = 0;
    return true;
  }

  if (in.src == in.dest) {
    GetOutput() = 1;
    return true;
  }

  auto buffer = InitBuffer(rank, in.src, in.count);
  RingTransfer(comm, rank, size, in.src, in.dest, buffer);

  GetOutput() = CheckAndReduce(comm, rank, in.dest, buffer);
  return true;
}

bool GalkinDRingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_ring
