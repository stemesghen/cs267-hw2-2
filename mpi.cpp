#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
// Put any static global variables here that you will use throughout the simulation.
std::vector<int> rank_start_index, rank_end_index;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); 

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int num_procs; //number of ranks (processors) 
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    rank_start_index.resize(num_procs);
    rank_end_index.resize(num_procs);

    init_simulation(parts, num_parts, size, my_rank, num_procs);

    for (int step = 0; step < num_steps; step++) {
        simulate_one_step(parts, num_parts, size, my_rank, num_procs);
        gather_for_save(parts, num_parts, size, my_rank, num_procs);
    }

    //Initialize seed - is that needed?  
    MPI_Finalize();
    return 0;

}

// Apply the force from a neighbor to a particle

void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;  // Ignore if beyond cutoff distance

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

    // Bounce off walls
    if (p.x < 0 || p.x > size) {
        p.x = (p.x < 0) ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    if (p.y < 0 || p.y > size) {
        p.y = (p.y < 0) ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    int particles_per_rank = num_parts / num_procs;
    if (rank < num_parts % num_procs) particles_per_rank++;

    rank_start_index[rank] = (rank == 0) ? 0 : rank_end_index[rank - 1];
    rank_end_index[rank] = rank_start_index[rank] + particles_per_rank;
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    //ranks start and end
    int particles_per_rank = num_parts / num_procs;
    if (rank < num_parts % num_procs) particles_per_rank++;

    int my_start = (rank == 0) ? 0 : rank_end_index[rank - 1];
    int my_end = my_start + particles_per_rank;

    int grid_size = ceil(size / cutoff);  //grid size

    std::vector<std::vector<int>> grid(grid_size * grid_size);
    for (int i = my_start; i < my_end; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;

        int grid_cell_x = floor(parts[i].x / cutoff);         //current grid cell of particle

        int grid_cell_y = floor(parts[i].y / cutoff);

        int grid_index = grid_cell_x + grid_cell_y * grid_size;  // create 1d from 2d
        grid[grid_index].push_back(i); 

    }



   // Exchange ghost particles with neighboring ranks
    std::vector<particle_t> ghost_particles_send, ghost_particles_recv;
    int left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_neighbor = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // Pack boundary particles for sending
    for (int i = my_start; i < my_end; i++) {
        if (parts[i].x < cutoff || parts[i].x > size - cutoff) {
            ghost_particles_send.push_back(parts[i]);
        }
    }

    // Send and receive ghost particles
    int send_count = ghost_particles_send.size();
    int recv_count;

    MPI_Sendrecv(&send_count, 1, MPI_INT, left_neighbor, 0, 
                &recv_count, 1, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    ghost_particles_recv.resize(recv_count);
    MPI_Sendrecv(ghost_particles_send.data(), send_count, MPI_PARTICLE, left_neighbor, 1, 
                ghost_particles_recv.data(), recv_count, MPI_PARTICLE, right_neighbor, 1, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Compute forces with both local and ghost particles
    for (int i = my_start; i < my_end; i++) {
        int grid_cell_x = floor(parts[i].x / cutoff);
        int grid_cell_y = floor(parts[i].y / cutoff);

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int neighbor_x = grid_cell_x + dx;
                int neighbor_y = grid_cell_y + dy;

                if (neighbor_x >= 0 && neighbor_x < grid_size &&
                    neighbor_y >= 0 && neighbor_y < grid_size) {
                    
                    int neighbor_index = neighbor_x + neighbor_y * grid_size;

                    for (int j : grid[neighbor_index]) {
                        if (i != j) {
                            apply_force(parts[i], parts[j]);
                        }
                    }
                }
            }
        }
    }

    // Compute forces for ghost particles
    for (auto &ghost : ghost_particles_recv) {
        for (int i = my_start; i < my_end; i++) {
            apply_force(parts[i], ghost);
        }
    }

        


   // Move Particles
    for (int i = my_start; i < my_end; i++) {
    
        move(parts[i], size);
    }


//Ghost Particles Needed
std::vector<particle_t> ghost_particles;




}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id. 
    MPI_Gather((rank == 0) ? MPI_IN_PLACE : parts + rank_start_index[rank],
               rank_end_index[rank] - rank_start_index[rank], MPI_PARTICLE,
               parts, rank_end_index[rank] - rank_start_index[rank], MPI_PARTICLE,
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




