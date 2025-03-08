#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

// Initialize particle positions and velocities
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    for (int i = 0; i < num_parts; i++) {
    parts[i].id = rank * num_parts + i; 
}

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

void move(particle_t& p, double size) {

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


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double region_size = size / num_procs;
    double bottom_bound = rank * region_size;
    double top_bound = (rank + 1) * region_size;

    std::vector<particle_t> send_top, send_bottom;
    std::vector<particle_t> recv_top, recv_bottom;
    std::vector<particle_t> real_particles;

    int grid_size = ceil(size / cutoff); 
    std::vector<std::vector<int>> grid(grid_size * grid_size);


    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);

        int grid_x = floor(parts[i].x / cutoff);
        int grid_y = floor(parts[i].y / cutoff);
        int grid_index = grid_x + grid_y * grid_size;

        if (grid_x >= 0 && grid_x < grid_size && grid_y >= 0 && grid_y < grid_size) {
            grid[grid_index].push_back(i);
        }

        // Identify ghost particles for MPI communication
        if (parts[i].y - bottom_bound <= cutoff) send_bottom.push_back(parts[i]);
        if (top_bound - parts[i].y <= cutoff) send_top.push_back(parts[i]);

        // Only store particles that belong to this process
        if (parts[i].y >= bottom_bound && parts[i].y <= top_bound) {
            real_particles.push_back(parts[i]);
        }
    }

    // Exchange ghost particles
    int bottom_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int top_rank = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    int send_counts[2] = {static_cast<int>(send_top.size()), static_cast<int>(send_bottom.size())};
    int recv_counts[2];

    MPI_Sendrecv(&send_counts[0], 1, MPI_INT, top_rank, 1,
                 &recv_counts[1], 1, MPI_INT, bottom_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&send_counts[1], 1, MPI_INT, bottom_rank, 2,
                 &recv_counts[0], 1, MPI_INT, top_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    recv_top.resize(recv_counts[0]);
    recv_bottom.resize(recv_counts[1]);


    MPI_Sendrecv(send_top.data(), send_counts[0], PARTICLE, top_rank, 3,
                 recv_bottom.data(), recv_counts[1], PARTICLE, bottom_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_bottom.data(), send_counts[1], PARTICLE, bottom_rank, 4,
                 recv_top.data(), recv_counts[0], PARTICLE, top_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //Reset Forces
    for (auto& p : real_particles) {
        p.ax = 0.0;
        p.ay = 0.0;

  }

    // Compute Forces Using Grid-Based Approach
    for (auto& p : real_particles) {
        int grid_x = floor(p.x / cutoff);
        int grid_y = floor(p.y / cutoff);
        int grid_index = grid_x + grid_y * grid_size;

        // Check the 3x3 neighboring grid cells
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int neighbor_x = grid_x + dx;
                int neighbor_y = grid_y + dy;

                if (neighbor_x >= 0 && neighbor_x < grid_size &&
                    neighbor_y >= 0 && neighbor_y < grid_size) {

                    int neighbor_index = neighbor_x + neighbor_y * grid_size;

                    for (int j : grid[neighbor_index]) {
                        if (p.id != parts[j].id) {
                            apply_force(p, parts[j]);
                        }
                    }
                }
            }
        }
    }

    //  Compute Forces for Ghost Particles
    for (auto &ghost : recv_top) {
        for (auto& p : real_particles) {
            apply_force(p, ghost);
        }
    }
    for (auto &ghost : recv_bottom) {
        for (auto& p : real_particles) {
            apply_force(p, ghost);
        }
    }

 // Move Particles After Force Computation
    for (size_t i = 0; i < real_particles.size(); i++) {
        move(real_particles[i], size);
    }

    // Copy Updated Particles Back to parts
    for (size_t i = 0; i < real_particles.size(); i++) {
        parts[i] = real_particles[i];
    }
}
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> recv_counts(num_procs);
    std::vector<int> displs(num_procs);

    int local_size = num_parts;


    // Gather local sizes
    MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<particle_t> all_particles;

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
