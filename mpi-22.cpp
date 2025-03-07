
#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

// Static global variables
std::vector<int> rank_start_index;
std::vector<int> rank_end_index;

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::random_device rd;
    std::mt19937 gen(rd() + rank);
    std::uniform_real_distribution<double> dist(0.0, size);

    double region_size = size / num_procs; // Each rank handles this portion of the domain

    std::vector<particle_t> local_particles;

    for (int i = 0; i < num_parts; i++) {
        particle_t p;
        p.x = dist(gen);
        p.y = dist(gen);
        p.vx = 0.0;
        p.vy = 0.0;
        p.ax = 0.0;
        p.ay = 0.0;
        p.id = i;

        // Determine which rank should own this particle
        int assigned_rank = std::min((int)(p.x / region_size), num_procs - 1);
        if (assigned_rank == rank) {
            local_particles.push_back(p);
        }
    }

    // Copy local particles to parts array
    std::copy(local_particles.begin(), local_particles.end(), parts);
}


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

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    //  Identify spatial domain boundaries
    double region_size = size / num_procs;
    double left_bound = rank * region_size;
    double right_bound = (rank + 1) * region_size;

    std::vector<particle_t> send_left, send_right;
    std::vector<particle_t> recv_left, recv_right;

    // 2️Identify particles near left/right boundaries
    for (int i = 0; i < num_parts; i++) {
        if (parts[i].x - left_bound < cutoff) send_left.push_back(parts[i]);
        if (right_bound - parts[i].x < cutoff) send_right.push_back(parts[i]);
    }

    // 3️Determine MPI neighbor ranks
    int left_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_rank = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    int send_left_size = send_left.size();
    int send_right_size = send_right.size();
    int recv_left_size = 0, recv_right_size = 0;

    // 4️Exchange ghost particle counts
    if (left_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(&send_right_size, 1, MPI_INT, right_rank, 0,
                     &recv_left_size, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (right_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(&send_left_size, 1, MPI_INT, left_rank, 0,
                     &recv_right_size, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Ensure no negative counts
    if (recv_left_size < 0 || recv_right_size < 0) {
        fprintf(stderr, "ERROR: Negative recv_size detected on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 5️Allocate memory for incoming ghost particles
    recv_left.resize(recv_left_size);
    recv_right.resize(recv_right_size);

    // 6️Exchange ghost particle data       
    if (left_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_left.data(), send_left_size * sizeof(particle_t), MPI_BYTE, left_rank, 1,
                     recv_right.data(), recv_right_size * sizeof(particle_t), MPI_BYTE, right_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (right_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_right.data(), send_right_size * sizeof(particle_t), MPI_BYTE, right_rank, 2,
                     recv_left.data(), recv_left_size * sizeof(particle_t), MPI_BYTE, left_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 7️Compute forces (include ghost particles)
    std::vector<particle_t> all_particles(parts, parts + num_parts);
    all_particles.insert(all_particles.end(), recv_left.begin(), recv_left.end());
    all_particles.insert(all_particles.end(), recv_right.begin(), recv_right.end());

    for (int i = 0; i < all_particles.size(); i++) {
        for (int j = 0; j < all_particles.size(); j++) {
            if (i != j) apply_force(all_particles[i], all_particles[j]);
        }
    }

    // 8️Move only local particles
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }
}



void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> recv_counts(num_procs), displs(num_procs);
    int local_size = rank_end_index[rank] - rank_start_index[rank];

    MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<particle_t> all_particles;
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
        all_particles.resize(displs[num_procs - 1] + recv_counts[num_procs - 1]);
    }

    MPI_Gatherv(parts, local_size * sizeof(particle_t), MPI_BYTE,
                all_particles.data(), recv_counts.data(), displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t &a, const particle_t &b) {
            return a.id < b.id;
        });
    }
}



