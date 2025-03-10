#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

extern MPI_Datatype PARTICLE;
std::vector<int> mpi_start_index;
std::vector<int> mpi_end_index;

void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;

    particle.ax += coef * dx;
    particle.ay += coef * dy;

    std::cout << "DEBUG: Applying force on P" << particle.id 
              << " due to P" << neighbor.id 
              << " → ax=" << particle.ax 
              << ", ay=" << particle.ay << std::endl;
}

void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

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
    std::cout << "Rank " << rank << " handling particles from " 
              << start_rank_index << " to " << end_rank_index << std::endl;
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
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
        int recv_count;
        MPI_Sendrecv(&send_count, 1, MPI_INT, rank + 1, 0, &recv_count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_bottom_ghosts.resize(recv_count);
        if (send_count > 0) {
            MPI_Sendrecv(top_ghost_particles.data(), send_count, PARTICLE, rank + 1, 1,
                         recv_bottom_ghosts.data(), recv_count, PARTICLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        std::cout << "Rank " << rank << " sent " << send_count << " particles to rank " << rank + 1
                  << " and received " << recv_count << " particles from below.\n";
    }

    if (rank > 0) {
        int send_count = bottom_ghost_particles.size();
        int recv_count;
        MPI_Sendrecv(&send_count, 1, MPI_INT, rank - 1, 0, &recv_count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_top_ghosts.resize(recv_count);
        if (send_count > 0) {
            MPI_Sendrecv(bottom_ghost_particles.data(), send_count, PARTICLE, rank - 1, 1,
                         recv_top_ghosts.data(), recv_count, PARTICLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        std::cout << "Rank " << rank << " sent " << send_count << " particles to rank " << rank - 1
                  << " and received " << recv_count << " particles from above.\n";
    }

    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
    }

    for (int i = 0; i < num_parts; ++i) {
        for (int j = i + 1; j < num_parts; ++j) {
            apply_force(parts[i], parts[j]);
            apply_force(parts[j], parts[i]);
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

    MPI_Gatherv(parts, num_parts, PARTICLE, all_particles.data(), send_counts.data(), displacements.data(), PARTICLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });

        std::cout << "Rank " << rank << " gathered " << all_particles.size() << " particles.\n";
    }
}
