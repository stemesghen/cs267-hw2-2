
#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;


// Put any static global variables here that you will use throughout the simulation.

extern MPI_Datatype PARTICLE;
 int num_cells_x;     // Number of grid cells along X
int num_cells_y;     // Number of grid cells along Y
 double domain_height; // Domain height



std::vector<int> mpi_start_index;
std::vector<int> mpi_end_index;



// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

    std::cout << "Applying force between particles:"
          << " P1(id=" << particle.id << ", x=" << particle.x << ", y=" << particle.y << ")"
          << " P2(id=" << neighbor.id << ", x=" << neighbor.x << ", y=" << neighbor.y << ")"
          << " Distance=" << sqrt(r2) 
          << " r2=" << r2
          << std::endl;

}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }

}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    mpi_start_index.resize(num_procs);
    mpi_end_index.resize(num_procs);

    int particles_per_rank = num_parts / num_procs;
    int remaining_particles = num_parts % num_procs;

    int start_rank_index, end_rank_index;
    if (rank < remaining_particles) {
        start_rank_index = rank * (particles_per_rank + 1);
        end_rank_index = start_rank_index + particles_per_rank + 1;
    } else {
        start_rank_index = rank * particles_per_rank + remaining_particles;
        end_rank_index = start_rank_index + particles_per_rank;
    }

    mpi_start_index[rank] = start_rank_index;
    mpi_end_index[rank] = end_rank_index;

    MPI_Barrier(MPI_COMM_WORLD);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    const double p_cutoff = 0.01;
    const int grid_size = ceil(size / p_cutoff);
    std::vector<std::vector<int>> grid(grid_size * grid_size);

    for (int i = 0; i < num_parts; i++) {
        int cell_x = floor(parts[i].x / p_cutoff);
        int cell_y = floor(parts[i].y / p_cutoff);
        grid[cell_x + cell_y * grid_size].push_back(i);
    }

    std::vector<particle_t> top_ghost_particles, bottom_ghost_particles;
    std::vector<particle_t> recv_top_ghosts, recv_bottom_ghosts;

    for (int i = mpi_start_index[rank]; i < mpi_end_index[rank]; i++) {
        if (parts[i].y >= (mpi_end_index[rank] - cutoff)) {
            top_ghost_particles.push_back(parts[i]);
        }
        if (parts[i].y <= (mpi_start_index[rank] + cutoff)) {
            bottom_ghost_particles.push_back(parts[i]);
        }
    }

    if (rank < num_procs - 1) {
        int send_count = top_ghost_particles.size();
        int recv_count = 0;

        MPI_Sendrecv(&send_count, 1, MPI_INT, rank + 1, 0,
                     &recv_count, 1, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (recv_count > 0) {
            recv_bottom_ghosts.resize(recv_count);
            MPI_Sendrecv(top_ghost_particles.data(), send_count, PARTICLE, rank + 1, 1,
                         recv_bottom_ghosts.data(), recv_count, PARTICLE, rank + 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if (rank > 0) {
        int send_count = bottom_ghost_particles.size();
        int recv_count = 0;

        MPI_Sendrecv(&send_count, 1, MPI_INT, rank - 1, 0,
                     &recv_count, 1, MPI_INT, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (recv_count > 0) {
            recv_top_ghosts.resize(recv_count);
            MPI_Sendrecv(bottom_ghost_particles.data(), send_count, PARTICLE, rank - 1, 1,
                         recv_top_ghosts.data(), recv_count, PARTICLE, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    for (int i = 0; i < num_parts; ++i) {
        int cell_x = (int)((parts[i].x + 1e-9) / cutoff);
        int cell_y = (int)((parts[i].y + 1e-9) / cutoff);

        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                int neighbor_x = cell_x + x;
                int neighbor_y = cell_y + y;

                if (neighbor_x >= 0 && neighbor_x < num_cells_x &&
                    neighbor_y >= 0 && neighbor_y < num_cells_y) {

                    int neighbor_idx = neighbor_x + neighbor_y * num_cells_x;

                    for (int j : grid[neighbor_idx]) {
                        if (i != j && j < num_parts) {
                            apply_force(parts[i], parts[j]);
                        }
                    }
                }
            }
        }
    }

    for (particle_t& ghost : recv_top_ghosts) {
        for (int i = 0; i < num_parts; ++i) {
            apply_force(parts[i], ghost);
        }
    }

    for (particle_t& ghost : recv_bottom_ghosts) {
        for (int i = 0; i < num_parts; ++i) {
            apply_force(parts[i], ghost);
        }
    }

    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }

    std::vector<particle_t> send_to_next, send_to_prev, new_local_particles;
    for (int i = 0; i < num_parts; i++) {
        if (parts[i].y > mpi_end_index[rank] && rank < num_procs - 1) {
            send_to_next.push_back(parts[i]);
        } else if (parts[i].y < mpi_start_index[rank] && rank > 0) {
            send_to_prev.push_back(parts[i]);
        } else {
            new_local_particles.push_back(parts[i]);
        }
    }

    std::vector<particle_t> recv_from_next, recv_from_prev;
    int send_count_next = send_to_next.size();
    int send_count_prev = send_to_prev.size();
    int recv_count_next = 0, recv_count_prev = 0;

    if (rank < num_procs - 1) {
        MPI_Sendrecv(&send_count_next, 1, MPI_INT, rank + 1, 0,
                     &recv_count_prev, 1, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank > 0) {
        MPI_Sendrecv(&send_count_prev, 1, MPI_INT, rank - 1, 0,
                     &recv_count_next, 1, MPI_INT, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (recv_count_next > 0) recv_from_next.resize(recv_count_next);
    if (recv_count_prev > 0) recv_from_prev.resize(recv_count_prev);

    if (rank < num_procs - 1 && recv_count_next > 0) {
        MPI_Recv(recv_from_next.data(), recv_count_next, PARTICLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < num_procs - 1 && send_count_next > 0) {
        MPI_Send(send_to_next.data(), send_count_next, PARTICLE, rank + 1, 1, MPI_COMM_WORLD);
    }
    if (rank > 0 && recv_count_prev > 0) {
        MPI_Recv(recv_from_prev.data(), recv_count_prev, PARTICLE, rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank > 0 && send_count_prev > 0) {
        MPI_Send(send_to_prev.data(), send_count_prev, PARTICLE, rank - 1, 2, MPI_COMM_WORLD);
    }

    new_local_particles.insert(new_local_particles.end(), recv_from_next.begin(), recv_from_next.end());
    new_local_particles.insert(new_local_particles.end(), recv_from_prev.begin(), recv_from_prev.end());

    MPI_Barrier(MPI_COMM_WORLD);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> send_counts(num_procs);
    MPI_Gather(&num_parts, 1, MPI_INT, send_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements(num_procs);
    int total_particles = 0;
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            displacements[i] = total_particles;
            total_particles += send_counts[i];
        }
    }

    std::vector<particle_t> all_particles;
    if (rank == 0) all_particles.resize(total_particles);

    MPI_Gatherv(parts, num_parts, PARTICLE,
                all_particles.data(), send_counts.data(), displacements.data(), PARTICLE,
                0, MPI_COMM_WORLD);
}