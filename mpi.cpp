
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







void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> send_counts(num_procs);
    MPI_Gather(&num_parts, 1, MPI_INT, send_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements(num_procs, 0);
    int total_particles = 0;
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            displacements[i] = total_particles;
            total_particles += send_counts[i];
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
        assert(mean_dist < 3e-7);
    }
}
