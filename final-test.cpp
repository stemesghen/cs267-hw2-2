
#include <cassert>
#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
using namespace std;


// Put any static global variables here that you will use throughout the simulation.

extern MPI_Datatype PARTICLE;
 int num_cells_x;     // Number of grid cells along X
int num_cells_y;     // Number of grid cells along Y
 double domain_height; // Domain height

std::vector<particle_t> my_particles;  // Declare globally


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

    // std::cout << "Applying force between particles:"
    //       << " P1(id=" << particle.id << ", x=" << particle.x << ", y=" << particle.y << ")"
    //       << " P2(id=" << neighbor.id << ", x=" << neighbor.x << ", y=" << neighbor.y << ")"
    //       << " Distance=" << sqrt(r2) 
    //       << " r2=" << r2
    //       << std::endl;

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
    // Resize vectors to store start and end indices for each rank
    mpi_start_index.resize(num_procs);
    mpi_end_index.resize(num_procs);

    double y_interval = size / num_procs;
    double starting_y = 0.0;

    // Assign start and end Y-coordinates for each processor
    for (int i = 0; i < num_procs; i++) {
        mpi_start_index[i] = starting_y;
        double y_cutoff = starting_y + y_interval;
        mpi_end_index[i] = y_cutoff;
        starting_y = y_cutoff;
    }

    // Check partitioning within mpi_start_index and mpi_end_index vectors
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            std::cout << "mpi_start_index[" << i << "] = " << mpi_start_index[i]
                      << ", mpi_end_index[" << i << "] = " << mpi_end_index[i] << std::endl;
        }
    }

    // Assign particles to the current rank based on Y-coordinate
    std::vector<particle_t> my_particles;
    for (int i = 0; i < num_parts; i++) {
        double y_coordinate = parts[i].y;

        if (y_coordinate > mpi_start_index[rank] && y_coordinate < mpi_end_index[rank]) {
my_particles.push_back(parts[i]);
        }
    }

    // Ensure all ranks synchronize before continuing
    MPI_Barrier(MPI_COMM_WORLD);
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    const double p_cutoff = 0.01;

    double top_boundary = mpi_end_index[rank] - p_cutoff;
    double bottom_boundary = mpi_start_index[rank] + p_cutoff;

    std::vector<particle_t> send_bottom_ghost_particles;
    std::vector<particle_t> send_top_ghost_particles;

    for (int i = 0; i < my_particles.size(); i++) {
        if (my_particles[i].y > top_boundary) {
            send_top_ghost_particles.push_back(my_particles[i]);
        }
        if (my_particles[i].y < bottom_boundary) {
            send_bottom_ghost_particles.push_back(my_particles[i]);
        }
    }

    int bottom_send_count = send_bottom_ghost_particles.size();
    int top_send_count = send_top_ghost_particles.size();
    int bottom_recv_count = 0, top_recv_count = 0;
 std::vector<particle_t> recv_top_ghost_particles;
    std::vector<particle_t> recv_bottom_ghost_particles;

    // Communication with neighbors
    if (rank > 0) {  // Has bottom neighbor
        MPI_Sendrecv(&bottom_send_count, 1, MPI_INT, rank - 1, 0,
                     &bottom_recv_count, 1, MPI_INT, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_bottom_ghost_particles.resize(bottom_recv_count);
        MPI_Sendrecv(send_bottom_ghost_particles.data(), bottom_send_count, PARTICLE, rank - 1, 1,
                     recv_bottom_ghost_particles.data(), bottom_recv_count, PARTICLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank < num_procs - 1) {  // Has top neighbor
        MPI_Sendrecv(&top_send_count, 1, MPI_INT, rank + 1, 0,
                     &top_recv_count, 1, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_top_ghost_particles.resize(top_recv_count);
        MPI_Sendrecv(send_top_ghost_particles.data(), top_send_count, PARTICLE, rank + 1, 1,
                     recv_top_ghost_particles.data(), top_recv_count, PARTICLE, rank + 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Apply forces between local particles
    for (int i = 0; i < my_particles.size(); i++) {
        for (int j = i + 1; j < my_particles.size(); j++) {
            apply_force(my_particles[i], my_particles[j]);
        }
    }

    // Apply forces from ghost particles
    for (int i = 0; i < my_particles.size(); i++) {
        for (int j = 0; j < recv_bottom_ghost_particles.size(); j++) {
            apply_force(my_particles[i], recv_bottom_ghost_particles[j]);
        }
        for (int j = 0; j < recv_top_ghost_particles.size(); j++) {
            apply_force(my_particles[i], recv_top_ghost_particles[j]);
        }
    }

    // Move local particles
    for (int i = 0; i < my_particles.size(); i++) {
        move(my_particles[i], size);
    }

    // REDISTRIBUTION: Move particles between ranks
    std::vector<particle_t> send_bottom_particles, send_top_particles;
    std::vector<int> remove_indices;

    for (int i = 0; i < my_particles.size(); i++) {
        if (my_particles[i].y < mpi_start_index[rank]) {
            send_bottom_particles.push_back(my_particles[i]);
            remove_indices.push_back(i);
        } else if (my_particles[i].y > mpi_end_index[rank]) {
            send_top_particles.push_back(my_particles[i]);
            remove_indices.push_back(i);
     }
    }

    // Remove particles **after** collecting indices
    for (int i = remove_indices.size() - 1; i >= 0; i--) {
        my_particles.erase(my_particles.begin() + remove_indices[i]);
    }

    int bottom_send_count_particles = send_bottom_particles.size();
    int top_send_count_particles = send_top_particles.size();
    int bottom_recv_count_particles = 0, top_recv_count_particles = 0;

    std::vector<particle_t> recv_bottom_particles;
    std::vector<particle_t> recv_top_particles;

    if (rank > 0) {
        MPI_Sendrecv(&bottom_send_count_particles, 1, MPI_INT, rank - 1, 0,
                     &bottom_recv_count_particles, 1, MPI_INT, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_bottom_particles.resize(bottom_recv_count_particles);
        MPI_Sendrecv(send_bottom_particles.data(), bottom_send_count_particles, PARTICLE, rank - 1, 1,
                     recv_bottom_particles.data(), bottom_recv_count_particles, PARTICLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank < num_procs - 1) {
        MPI_Sendrecv(&top_send_count_particles, 1, MPI_INT, rank + 1, 0,
                     &top_recv_count_particles, 1, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_top_particles.resize(top_recv_count_particles);
        MPI_Sendrecv(send_top_particles.data(), top_send_count_particles, PARTICLE, rank + 1, 1,
                     recv_top_particles.data(), top_recv_count_particles, PARTICLE, rank + 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Add received particles to my_particles
    my_particles.insert(my_particles.end(), recv_bottom_particles.begin(), recv_bottom_particles.end());
    my_particles.insert(my_particles.end(), recv_top_particles.begin(), recv_top_particles.end());



    MPI_Barrier(MPI_COMM_WORLD);  // Sync before redistribution
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> send_counts(num_procs);
    MPI_Gather(&num_parts, 1, MPI_INT, send_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements(num_procs, 0);
    int total_particles = 0;
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            displacements[i] = total_particles;
    }
  }

    std::vector<particle_t> all_particles;
    if (rank == 0) {
        all_particles.resize(total_particles);
    }

    MPI_Gatherv(parts, num_parts, PARTICLE,
                all_particles.data(), send_counts.data(), displacements.data(), PARTICLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });

        std::vector<double> avg_dists;
        for (size_t i = 0; i < std::min(50, static_cast<int>(all_particles.size())); i++) {
            double dist = sqrt(all_particles[i].x * all_particles[i].x +
                               all_particles[i].y * all_particles[i].y);
            avg_dists.push_back(dist);
        }

        double mean_dist = std::accumulate(avg_dists.begin(), avg_dists.end(), 0.0) / avg_dists.size();
        std::cout << "Checking assertion: mean_dist = " << mean_dist << std::endl;
        assert(mean_dist < 3e-7);
    }
}





