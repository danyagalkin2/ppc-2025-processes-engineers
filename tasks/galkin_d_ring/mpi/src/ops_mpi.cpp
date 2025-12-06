#include "galkin_d_ring/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

namespace galkin_d_ring {

GalkinDRingMPI::GalkinDRingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GalkinDRingMPI::ValidationImpl() {
  const auto &in = GetInput();
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const bool src_in_range = (0 <= in.src) && (in.src < size);
  const bool dest_in_range = (0 <= in.dest) && (in.dest < size);
  const bool count_ok = (in.count > 0);

  return src_in_range && dest_in_range && count_ok;
}

bool GalkinDRingMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool GalkinDRingMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto in = GetInput();
  const int src = in.src;
  const int dest = in.dest;
  const int count = in.count;

  // проверяем что номер источника/приёмника в диапазоне и есть что передавать
  const bool valid = (0 <= src && src < size) && (0 <= dest && dest < size) && (count > 0);

  if (!valid) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }
  // отправка самому себе
  if (src == dest) {
    GetOutput() = 1;
    return true;
  }

  // буфер для данных которые гоняем по кольцу
  std::vector<int> buffer(count);

  // источник инициализирует данные
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

  // обрабатываем только процессы, лежащие на пути от src до dest
  if (is_on_path) {
    const int prev = (rank - 1 + size) % size;
    const int next = (rank + 1) % size;

    MPI_Status status{};

    if (rank == src) {
      // источник запускает сообщение по кольцу
      MPI_Send(buffer.data(), count, MPI_INT, next, 0, MPI_COMM_WORLD);
    } else {
      // сначала принимаем данные от предыдущего соседа по кольцу
      MPI_Recv(buffer.data(), count, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);

      // все кроме приёмника пересылают сообщение дальше
      if (rank != dest) {
        MPI_Send(buffer.data(), count, MPI_INT, next, 0, MPI_COMM_WORLD);
      }
    }
  }

  // проверяем корректность только на процессе-приёмнике
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

  GetOutput() = global_ok;
  return true;
}

bool GalkinDRingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_ring
