#include "galkin_d_ring/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace galkin_d_ring {

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
  if (size < 1) {
    size = 1;
  }

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
  // Создаём отдельный коммуникатор "виртуальной топологии" (без Cart/Graph)
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  // RAII: гарантированно освободим коммуникатор при любом выходе из функции
  struct CommGuard {
    MPI_Comm *c;
    ~CommGuard() {
      if (c && *c != MPI_COMM_NULL) {
        MPI_Comm_free(c);
        *c = MPI_COMM_NULL;
      }
    }
  } guard{&comm};

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const auto in = GetInput();
  const int src = in.src;
  const int dest = in.dest;
  const int count = in.count;

  // Безопасность: одинаково на всех ранках
  if (!(count > 0) || src < 0 || src >= size || dest < 0 || dest >= size) {
    GetOutput() = 0;
    return true;
  }

  // Отправка самому себе допустима
  if (src == dest) {
    GetOutput() = 1;
    return true;
  }

  std::vector<int> buffer(count, 0);

  // Источник инициализирует данные
  if (rank == src) {
    for (int i = 0; i < count; ++i) {
      buffer[i] = i + 1;
    }
  }

  // Сколько шагов по часовой стрелке от src до dest
  const int steps = (dest - src + size) % size;

  // Проходим цепочкой: sender -> receiver, строго по кольцу, шаг за шагом
  for (int step = 1; step <= steps; ++step) {
    const int sender = (src + step - 1) % size;
    const int receiver = (src + step) % size;

    if (rank == sender) {
      MPI_Send(buffer.data(), count, MPI_INT, receiver, 0, comm);
    } else if (rank == receiver) {
      MPI_Recv(buffer.data(), count, MPI_INT, sender, 0, comm, MPI_STATUS_IGNORE);
    }
  }

  // Проверяем на dest
  int local_ok = 1;
  if (rank == dest) {
    for (int i = 0; i < count; ++i) {
      if (buffer[i] != i + 1) {
        local_ok = 0;
        break;
      }
    }
  }

  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, comm);

  GetOutput() = global_ok;
  return true;
}

bool GalkinDRingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_ring
