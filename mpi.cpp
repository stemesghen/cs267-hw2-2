#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>

// Apply force between two particles
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
}

void move(particle_t& p, double size) {
    p.x += p.vx * dt + 0.5 * p.ax * dt * dt;
    p.y += p.vy * dt + 0.5 * p.ay * dt * dt;
    p.vx += 0.5 * p.ax * dt;
    p.vy += 0.5 * p.ay * dt;

    if (p.x < 0 || p.x > size) {
        p.x = (p.x < 0) ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    if (p.y < 0 || p.y > size) {
        p.y = (p.y < 0) ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// Distribute particles based on position
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double region_size = size / num_procs;
    int local_count = 0;
    for (int i = 0; i < num_parts; i++) {
        int assigned_rank = std::min(num_procs - 1, static_cast<int>(parts[i].x / region_size));
        if (assigned_rank == rank) {
            parts[local_count++] = parts[i];
        }
    }
    // Resize to match the actual number of particles in this rank
    num_parts = local_count;
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double region_size = size / num_procs;

    // Identify ghost particles (neighbor exchange)
    std::vector<particle_t> send_left, send_right, recv_left, recv_right;
    int left_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    for (int i = 0; i < num_parts; i++) {
        if (parts[i].x < rank * region_size + cutoff) send_left.push_back(parts[i]);
        if (parts[i].x > (rank + 1) * region_size - cutoff) send_right.push_back(parts[i]);
    }

    int send_counts[2] = {static_cast<int>(send_left.size()), static_cast<int>(send_right.size())};
    int recv_counts[2];

    MPI_Request reqs[4];
    MPI_Irecv(&recv_counts[0], 1, MPI_INT, left_rank, 1, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&recv_counts[1], 1, MPI_INT, right_rank, 2, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(&send_counts[0], 1, MPI_INT, left_rank, 2, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&send_counts[1], 1, MPI_INT, right_rank, 1, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    recv_left.resize(recv_counts[0]);
    recv_right.resize(recv_counts[1]);

    MPI_Irecv(recv_left.data(), recv_counts[0], PARTICLE, left_rank, 3, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(recv_right.data(), recv_counts[1], PARTICLE, right_rank, 4, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(send_left.data(), send_counts[0], PARTICLE, left_rank, 4, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(send_right.data(), send_counts[1], PARTICLE, right_rank, 3, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Compute forces-local
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = parts[i].ay = 0.0;
        for (int j = 0; j < num_parts; j++) {
            if (i != j) {
                apply_force(parts[i], parts[j]);
            }
        }
    }

    // Compute forces-ghost 
    for (auto& ghost : recv_left) {
        for (int i = 0; i < num_parts; i++) {
            apply_force(parts[i], ghost);
        }
    }
    for (auto& ghost : recv_right) {
        for (int i = 0; i < num_parts; i++) {
            apply_force(parts[i], ghost);
        }
    }

    // Move particles
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> recv_counts(num_procs);
    std::vector<int> displs(num_procs);
    int local_size = num_parts;

    MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
    }

    MPI_Gatherv(parts, num_parts, PARTICLE,
                rank == 0 ? parts : nullptr, recv_counts.data(), displs.data(), PARTICLE,
                0, MPI_COMM_WORLD);
}



//MPI communication goes in here - 


//Communication between Rows: Use SendRecv  so that communication is more targeted 

// For Communication  - - consider ghost particles

//MPI_Sendrecv(&ghost_particles_send, send_count, MPI_PARTICLE, neighbor_rank,
             //0, &ghost_particles_recv, recv_count, MPI_PARTICLE, neighbor_rank, 0,
             //MPI_COMM_WORLD, MPI_STATUS);




// Write this function such that at the end of it, the master (rank == 0)

//MPI_AllReduce ? or Gather?




//Simulation Time = 0.08257 seconds for 1000 particles.


// 2.284e-06 seconds for 6000000 particles. - before serial code with 2 nodes

// 2.335e-06 seconds for 6000000 particles. - before sertial code with 1 node

//MPI_Finalize(); 




