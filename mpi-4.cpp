#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

// Initialize particle positions and velocities
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::random_device rd;
    std::mt19937 gen(rd() + rank);
    std::uniform_real_distribution<double> dist(0.0, size);

    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;
    }
}

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

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int grid_size = ceil(size / cutoff);
    std::vector<std::vector<int>> grid(grid_size);

//grid assigning
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;
        int grid_x = floor(parts[i].x / cutoff);

        if (grid_x >= 0 && grid_x < grid_size) {
            grid[grid_x].push_back(i);
        }
    }

    // Communication with neighbor in 1d (row layout top & bottom )
    std::vector<particle_t> send_top, send_bottom;
    std::vector<particle_t> recv_top, recv_bottom;

    int top_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int bottom_rank = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    for (int i = 0; i < num_parts; i++) {
        if (parts[i].y < cutoff) send_top.push_back(parts[i]);
        if (parts[i].y > size - cutoff) send_bottom.push_back(parts[i]);
    }

    int send_counts[2] = {static_cast<int>(send_top.size()),
                          static_cast<int>(send_bottom.size())};
    int recv_counts[2];

    MPI_Sendrecv(&send_counts[0], 1, MPI_INT, top_rank, 1,
                 &recv_counts[1], 1, MPI_INT, bottom_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&send_counts[1], 1, MPI_INT, bottom_rank, 2,
                 &recv_counts[0], 1, MPI_INT, top_rank, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    recv_top.resize(recv_counts[0]);
    recv_bottom.resize(recv_counts[1]);

    MPI_Sendrecv(send_top.data(), send_counts[0], PARTICLE, top_rank, 3,
                 recv_bottom.data(), recv_counts[1], PARTICLE, bottom_rank, 3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_bottom.data(), send_counts[1], PARTICLE, bottom_rank, 4,
                 recv_top.data(), recv_counts[0], PARTICLE, top_rank, 4,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // ghost partical combined
      std::vector<particle_t> all_particles(parts, parts + num_parts);
    all_particles.insert(all_particles.end(), recv_top.begin(), recv_top.end());
    all_particles.insert(all_particles.end(), recv_bottom.begin(), recv_bottom.end());

    for (int i = 0; i < all_particles.size(); i++) {
        for (int j = 0; j < all_particles.size(); j++) {
            if (i != j) apply_force(all_particles[i], all_particles[j]);
        }
    }

    for (int i = 0; i < num_parts; i++) {
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
