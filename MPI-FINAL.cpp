
#include <cassert>
#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
using namespace std;




// Define static global variables
extern MPI_Datatype PARTICLE;
int num_cells_x;     // Number of grid cells along X
int num_cells_y;     // Number of grid cells along Y
double domain_height; // Domain height 

std::vector<int> mpi_start_index;
std::vector<int> mpi_end_index;

void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
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
}

// Initialize simulation
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    domain_height = size;
    mpi_start_index.resize(num_procs);
    mpi_end_index.resize(num_procs);

    double y_start = rank * (domain_height / num_procs);
    double y_end = (rank + 1) * (domain_height / num_procs);

    // Assign particles to this rank
    std::vector<particle_t> local_particles;
    for (int i = 0; i < num_parts; ++i) {
        if (parts[i].y >= y_start && parts[i].y < y_end) {
            local_particles.push_back(parts[i]);
        }
    }

    // Copy local particles back
    for (size_t i = 0; i < local_particles.size(); ++i) {
        parts[i] = local_particles[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Simulate one step
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double y_start = rank * (domain_height / num_procs);
    double y_end = (rank + 1) * (domain_height / num_procs);
    double boundary_tolerance = 1e-7;

    std::vector<particle_t> top_ghost_particles, bottom_ghost_particles;
    std::vector<particle_t> recv_top_ghosts, recv_bottom_ghosts;

    for (int i = 0; i < num_parts; i++) {
        if (parts[i].y >= (y_end - cutoff - boundary_tolerance)) {
            top_ghost_particles.push_back(parts[i]);
        }
        if (parts[i].y <= (y_start + cutoff + boundary_tolerance)) {
            bottom_ghost_particles.push_back(parts[i]);
        }
    }

    // MPI communication: Send/Receive ghost particles
    if (num_procs > 1) {
        int send_count_up = top_ghost_particles.size();
        int send_count_down = bottom_ghost_particles.size();
        int recv_count_up = 0, recv_count_down = 0;

        if (rank < num_procs - 1) {
            MPI_Sendrecv(&send_count_up, 1, MPI_INT, rank + 1, 0,
                         &recv_count_up, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_bottom_ghosts.resize(recv_count_up);
            MPI_Sendrecv(top_ghost_particles.data(), send_count_up, PARTICLE, rank + 1, 1,
                         recv_bottom_ghosts.data(), recv_count_up, PARTICLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank > 0) {
            MPI_Sendrecv(&send_count_down, 1, MPI_INT, rank - 1, 0,
                         &recv_count_down, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_top_ghosts.resize(recv_count_down);
            MPI_Sendrecv(bottom_ghost_particles.data(), send_count_down, PARTICLE, rank - 1, 1,
                         recv_top_ghosts.data(), recv_count_down, PARTICLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Reset forces
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = parts[i].ay = 0;
    }

    // Apply forces (particles within processor's region)
    for (int i = 0; i < num_parts; i++) {
        for (int j = i + 1; j < num_parts; j++) {
            apply_force(parts[i], parts[j]);
            apply_force(parts[j], parts[i]);
        }
    }
  // Apply forces with ghost particles
    for (particle_t& ghost : recv_top_ghosts) {
        for (int i = 0; i < num_parts; i++) {
            apply_force(parts[i], ghost);
        }
    }

    for (particle_t& ghost : recv_bottom_ghosts) {
        for (int i = 0; i < num_parts; i++) {
            apply_force(parts[i], ghost);
        }
    }

    // Move particles
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Sync before redistribution
}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int local_num_parts = num_parts; // Each rank only knows its local number of particles

    // Gather the local particle counts to rank 0
    std::vector<int> recv_counts(num_procs, 0);
    MPI_Gather(&local_num_parts, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);


    std::vector<int> displacements(num_procs, 0);
    int total_particles = 0;
    if (rank == 0) {
        for (int i = 1; i < num_procs; i++) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
        total_particles = displacements[num_procs - 1] + recv_counts[num_procs - 1];
    }

    // space for all particles on rank 0
    std::vector<particle_t> all_particles;
    if (rank == 0) {
        all_particles.resize(total_particles);
    }

    // Gather particles from ALL the processors
    MPI_Gatherv(parts, local_num_parts, PARTICLE,
                rank == 0 ? all_particles.data() : nullptr,
                recv_counts.data(), displacements.data(), PARTICLE,
                0, MPI_COMM_WORLD);


    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });
    }
}




