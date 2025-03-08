#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstddef> 

void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    double coef = (1.0 / pow(r, 12)) / mass;

    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t& p, double size) {
    p.x += p.vx * dt + 0.5 * p.ax * dt * dt;
    p.y += p.vy * dt + 0.5 * p.ay * dt * dt;

    p.vx += 0.5 * p.ax * dt;
    p.vy += 0.5 * p.ay * dt;

    while (p.x < 0 || p.x > size) {
        p.x = (p.x < 0) ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = (p.y < 0) ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int base_count = num_parts / num_procs;
    int extra = num_parts % num_procs;
    int my_start = rank * base_count + std::min(rank, extra);
    int my_end = my_start + base_count + (rank < extra);

    for (int i = my_start; i < my_end; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int base_count = num_parts / num_procs;
    int extra = num_parts % num_procs;
    int my_start = rank * base_count + std::min(rank, extra);
    int my_end = my_start + base_count + (rank < extra);

    int grid_size = ceil(size / cutoff);
    std::vector<std::vector<int>> grid(grid_size * grid_size);

    for (int i = my_start; i < my_end; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;
        int grid_x = floor(parts[i].x / cutoff);
        int grid_y = floor(parts[i].y / cutoff);

        if (grid_x >= 0 && grid_x < grid_size && grid_y >= 0 && grid_y < grid_size) {
            int grid_index = grid_x + grid_y * grid_size;
            grid[grid_index].push_back(i);
        }
    }

    std::vector<particle_t> ghost_particles_send, ghost_particles_recv;
    int top_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int bottom_neighbor = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    for (int i = my_start; i < my_end; i++) {
        if (parts[i].y < cutoff || parts[i].y > size - cutoff) {
            ghost_particles_send.push_back(parts[i]);
        }
    }

    int send_count = ghost_particles_send.size();
    int recv_count_top, recv_count_bottom;

    MPI_Sendrecv(&send_count, 1, MPI_INT, top_neighbor, 0,
                 &recv_count_bottom, 1, MPI_INT, bottom_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&send_count, 1, MPI_INT, bottom_neighbor, 1,
                 &recv_count_top, 1, MPI_INT, top_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    ghost_particles_recv.resize(recv_count_top + recv_count_bottom);
    MPI_Sendrecv(ghost_particles_send.data(), send_count, MPI_PARTICLE, top_neighbor, 2,
                 ghost_particles_recv.data(), recv_count_bottom, MPI_PARTICLE, bottom_neighbor, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(ghost_particles_send.data(), send_count, MPI_PARTICLE, bottom_neighbor, 3,
                 ghost_particles_recv.data() + recv_count_bottom, recv_count_top, MPI_PARTICLE, top_neighbor, 3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = my_start; i < my_end; i++) {
        int grid_x = floor(parts[i].x / cutoff);
        int grid_y = floor(parts[i].y / cutoff);

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int neighbor_x = grid_x + dx;
                int neighbor_y = grid_y + dy;

                if (neighbor_x >= 0 && neighbor_x < grid_size &&
                    neighbor_y >= 0 && neighbor_y < grid_size) {
                    int neighbor_index = neighbor_x + neighbor_y * grid_size;
                    for (int j : grid[neighbor_index]) {
                        if (i != j) {
                            apply_force(parts[i], parts[j]);
                        }
                    }
                }
            }
        }
    }

    for (auto &ghost : ghost_particles_recv) {
        for (int i = my_start; i < my_end; i++) {
            apply_force(parts[i], ghost);
        }
    }

    for (int i = my_start; i < my_end; i++) {
        move(parts[i], size);
    }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> recv_counts(num_procs);
    std::vector<int> displs(num_procs);

    int local_size = num_parts;

    // Gather local sizes
    MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<particle_t> all_particles; // Declare here to fix error

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
        all_particles.resize(displs[num_procs - 1] + recv_counts[num_procs - 1]); // Allocate space
    }

    // Gather particles to rank 0
    MPI_Gatherv(parts, num_parts, PARTICLE,
                rank == 0 ? all_particles.data() : nullptr, recv_counts.data(), displs.data(), PARTICLE,
                0, MPI_COMM_WORLD);

    // Sort only on rank 0
    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t &a, const particle_t &b) {
            return a.id < b.id;
        });
    }
}
