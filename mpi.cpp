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
    parts[i].id = rank * num_parts + i; // Assign unique ID based on MPI rank
}

    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;
    }
}



void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distancecd 
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
    double region_size = size / num_procs;
    double bottom_bound = rank * region_size;
    double top_bound = (rank + 1) * region_size;

    std::vector<particle_t> send_top, send_bottom;
    std::vector<particle_t> recv_top, recv_bottom;

    std::vector<particle_t> real_particles;

    for (int i = 0; i < num_parts; i++) {
        if (parts[i].y - bottom_bound <= cutoff && rank > 0) send_bottom.push_back(parts[i]);  // Close to bottom
        if (top_bound - parts[i].y <= cutoff && rank < num_procs -1) send_top.push_back(parts[i]);        // Close to top

        if (parts[i].y >= bottom_bound && parts[i].y <= top_bound) {
            real_particles.push_back(parts[i]);  // Only local (owned) particles move
       }
    }

    // ranks for top and bottom neighbors
    int bottom_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int top_rank = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Exchange ghost Y-direction
    int send_counts_tb[2] = {static_cast<int>(send_top.size()), static_cast<int>(send_bottom.size())};
    int recv_counts_tb[2];

    MPI_Sendrecv(&send_counts_tb[0], 1, MPI_INT, top_rank, 1,
                 &recv_counts_tb[1], 1, MPI_INT, bottom_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&send_counts_tb[1], 1, MPI_INT, bottom_rank, 2,
                 &recv_counts_tb[0], 1, MPI_INT, top_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    recv_top.resize(recv_counts_tb[0]);
    recv_bottom.resize(recv_counts_tb[1]);

    // Exchange ghost  Y-direction
    MPI_Sendrecv(send_top.data(), send_counts_tb[0], PARTICLE, top_rank, 3,
                 recv_bottom.data(), recv_counts_tb[1], PARTICLE, bottom_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_bottom.data(), send_counts_tb[1], PARTICLE, bottom_rank, 4,
                 recv_top.data(), recv_counts_tb[0], PARTICLE, top_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<particle_t> all_particles = real_particles;  // Start with owned particles
    all_particles.insert(all_particles.end(), recv_top.begin(), recv_top.end());  // Add ghosts
    all_particles.insert(all_particles.end(), recv_bottom.begin(), recv_bottom.end());  // Add ghosts

    // Debugging
    std::cout << "Rank " << rank << " sending " << send_top.size() << " particles to top" << std::endl;
    std::cout << "Rank " << rank << " sending " << send_bottom.size() << " particles to bottom" << std::endl;
std::cout << "Rank " << rank << " receiving " << recv_top.size() << " ghost particles from top" << std::endl;
    std::cout << "Rank " << rank << " receiving " << recv_bottom.size() << " ghost particles from bottom" << std::endl;


    for (auto& p : real_particles) {
        p.ax = 0.0;
        p.ay = 0.0;
    }




    for (auto& p : real_particles) {
        for (auto& neighbor : all_particles) {
            if (p.id == neighbor.id) continue;  // Ensure no self-interaction
            apply_force(p, neighbor);
        }
    }

        // **Debugging Forces Applied**
    for (auto& p : real_particles) {
        std::cout << "Rank " << rank << " particle " << p.id
                  << " ax=" << p.ax << " ay=" << p.ay << std::endl;
    }

    for (auto& p : real_particles) {
        move(p, size);
    }

    for (size_t i = 0; i < real_particles.size(); i++) {
        parts[i] = real_particles[i];
    }
}



// Gather particle data to the root rank for saving
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> recv_counts(num_procs), displs(num_procs);
    int local_size = num_parts;

    MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<particle_t> all_particles;

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
        all_particles.resize(displs[num_procs - 1] + recv_counts[num_procs - 1]);
    }

    MPI_Gatherv(parts, num_parts, PARTICLE,
                rank == 0 ? all_particles.data() : nullptr, recv_counts.data(), displs.data(), PARTICLE,
                0, MPI_COMM_WORLD);

     if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t &a, const particle_t &b) {
            return a.id < b.id;
        });
    }

}
