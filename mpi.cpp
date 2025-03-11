

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
   
   	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here



    /* Initialize based on location of all the particle and
     how you split all the particles into different rows


     Each processor handles a row and the particles that are in that row

     Each processor also simulates their particles movement per step. 

     */
       // Resize vectors to store start and end indices for each rank

    mpi_start_index.resize(num_procs);
    mpi_end_index.resize(num_procs);

    // Number of particles to assign per processor
    int particles_per_rank = num_parts / num_procs;
    int remaining_particles = num_parts % num_procs;

    // Calculate the start and end indices for the particles assigned to this rank
    int start_rank_index, end_rank_index;
    if (rank < remaining_particles) {
        // Distribute the remaining particles among the first 'remaining_particles' ranks
        start_rank_index = rank * (particles_per_rank + 1);
        end_rank_index = start_rank_index + particles_per_rank;
    } else {
        // Distribute the particles evenly among the remaining ranks
        start_rank_index = rank * particles_per_rank + remaining_particles;
        end_rank_index = start_rank_index + particles_per_rank - 1;
    }

    // Store the start and end indices for this rank
    mpi_start_index[rank] = start_rank_index;
    mpi_end_index[rank] = end_rank_index;

    // Perform domain decomposition (assign particles to processors based on their y-coordinate)
    double y_start = rank * (domain_height / num_procs);
    double y_end = (rank + 1) * (domain_height / num_procs);

    // Assign particles to this rank based on their position
    std::vector<particle_t> local_particles;
    for (int i = 0; i < num_parts; ++i) {
        // Check if the particle's position falls within the rank's region
        if (parts[i].y >= y_start && parts[i].y < y_end) {
            local_particles.push_back(parts[i]);
        }
    }

    // Copy local particles back to the original array
    for (size_t i = 0; i < local_particles.size(); ++i) {
        parts[i] = local_particles[i];
    }

    // Ensure all ranks synchronize before continuing
    MPI_Barrier(MPI_COMM_WORLD);  
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    /*
    
    Runs simulation per process 
    May need to redistribute the particles within this function 
    Because the next row processor has to have the correct set of particles to sumulate
    Some particles may go across the different processors after each step. 

    Communication -  ONLY for ghost particles before force calculation happens: 
        Communication happens if a particls is in row A & another particle in row A+ 1 
            -  they are close within cut off distance, then they will have force affecting each other. 

            you need to consider particls outside that row (in the top or bottom neighor ranks)
            Limit communication: 
                - so use point to point communication (Send Recv with that neighbor)
                - dynamic array or list or another data structure to consider the particles within the cutoff range
                - REMEMBER: storing redundant data BUT it avoids communication. 


    
    */
    
    
    
    const double p_cutoff = 0.01;
    const int grid_size = ceil(size / p_cutoff); 
    std::vector<std::vector<int>> grid(grid_size * grid_size); // grid cells with particle indices

    // Assign particles to grid cells (based on position)
    for (int i = 0; i < num_parts; i++) {
        int cell_x = floor(parts[i].x / p_cutoff);
        int cell_y = floor(parts[i].y / p_cutoff);
        grid[cell_x + cell_y * grid_size].push_back(i);
    }

    // Prepare ghost particles for communication
    std::vector<particle_t> top_ghost_particles, bottom_ghost_particles;
    std::vector<particle_t> recv_top_ghosts, recv_bottom_ghosts;

    double y_start = rank * (domain_height / num_procs);
    double y_end = (rank + 1) * (domain_height / num_procs);
    double boundary_tolerance = 1e-7;

    // Identify ghost particles based on the position
    for (int i = mpi_start_index[rank]; i < mpi_end_index[rank]; i++) {
        if (parts[i].y >= (y_end - cutoff - boundary_tolerance)) {
            top_ghost_particles.push_back(parts[i]);
        }
        if (parts[i].y <= (y_start + cutoff + boundary_tolerance)) {
            bottom_ghost_particles.push_back(parts[i]);
        }
    }

    // Exchange ghost particles based on rank
    if (rank == 0) {
        if (num_procs > 1) {
            int send_count = top_ghost_particles.size();
            int recv_count = 0;
            MPI_Sendrecv(&send_count, 1, MPI_INT, rank + 1, 0,
                         &recv_count, 1, MPI_INT, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (recv_count > 0) recv_bottom_ghosts.resize(recv_count);
            if (send_count > 0) {
                MPI_Sendrecv(top_ghost_particles.data(), send_count, PARTICLE, rank + 1, 1,
                             recv_bottom_ghosts.data(), recv_count, PARTICLE, rank + 1, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else if (rank == num_procs - 1) {
        if (num_procs > 1) {
            int send_count = bottom_ghost_particles.size();
            int recv_count = 0;
            MPI_Sendrecv(&send_count, 1, MPI_INT, rank - 1, 0,
                         &recv_count, 1, MPI_INT, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (recv_count > 0) recv_top_ghosts.resize(recv_count);
            if (send_count > 0) {
                MPI_Sendrecv(bottom_ghost_particles.data(), send_count, PARTICLE, rank - 1, 1,
                             recv_top_ghosts.data(), recv_count, PARTICLE, rank - 1, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        int send_count_up = top_ghost_particles.size();
        int send_count_down = bottom_ghost_particles.size();
        int recv_count_up = 0;
        int recv_count_down = 0;

        MPI_Sendrecv(&send_count_up, 1, MPI_INT, rank + 1, 0,
                     &recv_count_up, 1, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&send_count_down, 1, MPI_INT, rank - 1, 0,
                     &recv_count_down, 1, MPI_INT, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (recv_count_up > 0) recv_bottom_ghosts.resize(recv_count_up);
        if (recv_count_down > 0) recv_top_ghosts.resize(recv_count_down);

        if (send_count_up > 0) {
            MPI_Sendrecv(top_ghost_particles.data(), send_count_up, PARTICLE, rank + 1, 1,
                         recv_bottom_ghosts.data(), recv_count_up, PARTICLE, rank + 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (send_count_down > 0) {
            MPI_Sendrecv(bottom_ghost_particles.data(), send_count_down, PARTICLE, rank - 1, 1,
                         recv_top_ghosts.data(), recv_count_down, PARTICLE, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Force calculation (after ghost particles are exchanged)
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
    }

    // Apply force between particles and ghosts
    for (int i = 0; i < num_parts; ++i) {
        for (int j = i + 1; j < num_parts; ++j) {
            apply_force(parts[i], parts[j]);
            apply_force(parts[j], parts[i]);
        }
    }

    for (particle_t& ghost : recv_top_ghosts) {
        for (int i = 0; i < num_parts; ++i) {
            apply_force(parts[i], ghost);
            apply_force(ghost, parts[i]);
        }
    }

    for (particle_t& ghost : recv_bottom_ghosts) {
        for (int i = 0; i < num_parts; ++i) {
            apply_force(parts[i], ghost);
            apply_force(ghost, parts[i]);
        }
    }

    // Move particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }


/// REDISTRIBUTION 
    // After moving the particles, check if they have crossed boundaries and update ghost particles
    std::vector<particle_t> moved_top_ghost_particles, moved_bottom_ghost_particles;

    for (int i = mpi_start_index[rank]; i < mpi_end_index[rank]; i++) {
        if (parts[i].y >= (y_end - cutoff - boundary_tolerance)) {
            moved_top_ghost_particles.push_back(parts[i]);
        }
        if (parts[i].y <= (y_start + cutoff + boundary_tolerance)) {
            moved_bottom_ghost_particles.push_back(parts[i]);
        }
    }

    // If there are multiple ranks, exchange ghost particles
    if (num_procs > 1) {
        // For Rank 0
        if (rank == 0 && !moved_top_ghost_particles.empty()) {
            MPI_Sendrecv(moved_top_ghost_particles.data(), moved_top_ghost_particles.size(), PARTICLE,
                         rank + 1, 0, recv_bottom_ghosts.data(), moved_top_ghost_particles.size(),
                         PARTICLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // For Rank num_procs - 1 (last rank)
        else if (rank == num_procs - 1 && !moved_bottom_ghost_particles.empty()) {
            MPI_Sendrecv(moved_bottom_ghost_particles.data(), moved_bottom_ghost_particles.size(), PARTICLE,
                         rank - 1, 0, recv_top_ghosts.data(), moved_bottom_ghost_particles.size(),
                         PARTICLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // For Middle Ranks (between Rank 0 and num_procs - 1)
        else if (rank > 0 && rank < num_procs - 1) {
            // Exchange ghost particles with both neighbors
            MPI_Sendrecv(moved_top_ghost_particles.data(), moved_top_ghost_particles.size(), PARTICLE,
                         rank + 1, 0, recv_bottom_ghosts.data(), moved_top_ghost_particles.size(),
                         PARTICLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(moved_bottom_ghost_particles.data(), moved_bottom_ghost_particles.size(), PARTICLE,
                         rank - 1, 0, recv_top_ghosts.data(), moved_bottom_ghost_particles.size(),
                         PARTICLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  // Sync before redistribution
}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int num_cells_x = sqrt(num_procs);
    int num_cells_y = num_procs / num_cells_x;

    double cell_width = size / num_cells_x;
    double cell_height = size / num_cells_y;

    std::vector<int> send_counts(num_procs, 0);
    std::vector<int> recv_counts(num_procs, 0);
    std::vector<int> displacements(num_procs, 0);
    std::vector<std::vector<particle_t>> partitioned_particles(num_procs);

    for (int i = 0; i < num_parts; i++) {
        int cell_x = std::min(static_cast<int>(parts[i].x / cell_width), num_cells_x - 1);
        int cell_y = std::min(static_cast<int>(parts[i].y / cell_height), num_cells_y - 1);
        int target_rank = cell_x + cell_y * num_cells_x;  // Compute MPI rank based on grid position
        partitioned_particles[target_rank].push_back(parts[i]);
    }

    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = partitioned_particles[i].size();
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_particles = 0;
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            displacements[i] = total_particles;
            total_particles += recv_counts[i];  
        }
    }

    std::vector<particle_t> send_buffer;
    for (int i = 0; i < num_procs; i++) {
        send_buffer.insert(send_buffer.end(), partitioned_particles[i].begin(), partitioned_particles[i].end());
    }

    std::vector<particle_t> all_particles;
    if (rank == 0) {
        all_particles.resize(total_particles);
    }

    MPI_Gatherv(send_buffer.data(), send_buffer.size(), PARTICLE,
                rank == 0 ? all_particles.data() : nullptr, recv_counts.data(), displacements.data(), PARTICLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t& a, const particle_t& b) {
            return (a.y < b.y) || (a.y == b.y && a.x < b.x);
        });

        std::vector<double> avg_dists;
        size_t num_avg = std::min<size_t>(50, all_particles.size());

        for (size_t i = 0; i < num_avg; i++) {
            double dist = sqrt(all_particles[i].x * all_particles[i].x +
                               all_particles[i].y * all_particles[i].y);
            avg_dists.push_back(dist);
        }

        double mean_dist = (num_avg > 0) ? std::accumulate(avg_dists.begin(), avg_dists.end(), 0.0) / num_avg : 0.0;
        std::cout << "Checking assertion: mean_dist = " << mean_dist << std::endl;

        if (num_avg > 0) {
            assert(mean_dist < 3e-7);
        }
    }
}
