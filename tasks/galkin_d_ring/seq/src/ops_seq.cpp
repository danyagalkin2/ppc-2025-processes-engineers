#include "galkin_d_ring/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <vector>

namespace galkin_d_ring {

GalkinDRingSEQ::GalkinDRingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GalkinDRingSEQ::ValidationImpl() {
  const auto &in = GetInput();

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const bool src_in_range = (0 <= in.src) && (in.src < size);
  const bool dest_in_range = (0 <= in.dest) && (in.dest < size);
  const bool count_ok = (in.count > 0);

  return src_in_range && dest_in_range && count_ok;
}

bool GalkinDRingSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool GalkinDRingSEQ::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto in = GetInput();
  const int src = in.src;
  const int dest = in.dest;
  const int count = in.count;

  const bool valid = (0 <= src && src < size) && (0 <= dest && dest < size) && (count > 0);

  if (!valid) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  if (src == dest) {
    GetOutput() = 1;
    return true;
  }

  MPI_Comm ring_comm = MPI_COMM_NULL;
  int dims[1] = {size};
  int periods[1] = {1};
  const int reorder = 0;

  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &ring_comm);
  if (ring_comm == MPI_COMM_NULL) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  std::vector<int> buffer(count);
  if (rank == src) {
    for (int i = 0; i < count; ++i) {
      buffer[i] = i + 1;
    }
  }

  // расстояние по кольцу по часовой стрелке от src до dest
  const int clockwise_distance = (dest - src + size) % size;

  // положение текущего процесса относительно src по кольцу
  const int offset_from_src = (rank - src + size) % size;
  const bool is_on_path = (offset_from_src <= clockwise_distance);

  if (is_on_path) {
    // берём соседей
    int prev_cart = MPI_PROC_NULL;
    int next_cart = MPI_PROC_NULL;
    MPI_Cart_shift(ring_comm, 0, 1, &prev_cart, &next_cart);

    MPI_Status status{};

    if (rank == src) {
      // источник запускает сообщение по кольцу
      MPI_Send(buffer.data(), count, MPI_INT, next_cart, 0, ring_comm);
    } else {
      // сначала принимаем от предыдущего соседа по кольцу
      MPI_Recv(buffer.data(), count, MPI_INT, prev_cart, 0, ring_comm, &status);

      // все кроме приёмника пересылают дальше
      if (rank != dest) {
        MPI_Send(buffer.data(), count, MPI_INT, next_cart, 0, ring_comm);
      }
    }
  }

  int local_ok = 1;
  if (rank == dest) {
    for (int i = 0; i < count; ++i) {
      const int expected = i + 1;
      if (buffer[i] != expected) {
        local_ok = 0;
        break;
      }
    }
  }

  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

  MPI_Comm_free(&ring_comm);

  GetOutput() = global_ok;

  return true;
}

bool GalkinDRingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_ring
