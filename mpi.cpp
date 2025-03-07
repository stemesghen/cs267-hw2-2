#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>

// Initialize particle positions and assign to ranks
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double region_size = size / num_procs;  // Divide domain equally
    std::vector<particle_t> local_particles;

    for (int i = 0; i < num_parts; i++) {
        if ((parts[i].x / region_size) == rank) {
            local_particles.push_back(parts[i]);  // Assign particles based on position
        }
    }
    std::copy(local_particles.begin(), local_particles.end(), parts);
}

// Apply force between two particles
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

// Move the particle using Velocity Verlet integration
void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    if (p.x < 0 || p.x > size) {
        p.x = (p.x < 0) ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    if (p.y < 0 || p.y > size) {
        p.y = (p.y < 0) ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// Simulation step with domain decomposition
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double region_size = size / num_procs;
    double left_bound = rank * region_size;
    double right_bound = (rank + 1) * region_size;

    std::vector<particle_t> send_left, send_right;
    std::vector<particle_t> recv_left, recv_right;

    for (int i = 0; i < num_parts; i++) {
        if (parts[i].x - left_bound < cutoff) send_left.push_back(parts[i]);
        if (right_bound - parts[i].x < cutoff) send_right.push_back(parts[i]);
    }

    int left_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_rank = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    int send_left_size = send_left.size(), send_right_size = send_right.size();
    int recv_left_size = 0, recv_right_size = 0;

    if (left_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(&send_right_size, 1, MPI_INT, right_rank, 0,
                     &recv_left_size, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (right_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(&send_left_size, 1, MPI_INT, left_rank, 0,
                     &recv_right_size, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    recv_left.resize(recv_left_size);
    recv_right.resize(recv_right_size);

    if (left_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_left.data(), send_left_size, PARTICLE, left_rank, 1,
                     recv_right.data(), recv_right_size, PARTICLE, right_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (right_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_right.data(), send_right_size, PARTICLE, right_rank, 2,
                     recv_left.data(), recv_left_size, PARTICLE, left_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::vector<particle_t> all_particles(parts, parts + num_parts);
    all_particles.insert(all_particles.end(), recv_left.begin(), recv_left.end());
    all_particles.insert(all_particles.end(), recv_right.begin(), recv_right.end());

    for (size_t i = 0; i < all_particles.size(); i++) {
        for (size_t j = i + 1; j < all_particles.size(); j++) {
            apply_force(all_particles[i], all_particles[j]);
        }
    }

    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }
}

// Gather results
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> recv_counts(num_procs), displs(num_procs);

    // Gather the number of particles from each process
    MPI_Gather(&num_parts, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
    }

    std::vector<particle_t> all_parts(rank == 0 ? std::accumulate(recv_counts.begin(), recv_counts.end(), 0) : 0);

    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : parts, num_parts, PARTICLE,
                rank == 0 ? all_parts.data() : nullptr, recv_counts.data(), displs.data(), PARTICLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::sort(all_parts.begin(), all_parts.end(), [](const particle_t &a, const particle_t &b) {
            return a.id < b.id;
        });

        std::copy(all_parts.begin(), all_parts.end(), parts);
    }
}
