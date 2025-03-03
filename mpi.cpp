#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
// Put any static global variables here that you will use throughout the simulation.
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); 

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int num_procs; //number of ranks (processors) 
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    rank_start_index.resize(num_procs);
    rank_end_index.resize(num_procs);

    //Initialize seed - is that needed?  

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
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {


    int current_rank_start_index = 0;

    //loop over ranks
    for (int irank = 0; irank < num_procs; irank++) {
            int particles_per_rank = num_parts / num_procs;

            if (rank < num_parts % num_procs) particles_per_rank++;

            rank_start_index[irank] = current_rank_start_index;
            rank_end_index[irank] = current_rank_start_index + nparticles_per_rank;

            current_rank_start_index += nparticles_per_rank;


            int my_start = (rank == 0) ? 0 : rank_end_index[rank - 1];
            int my_end = rank_start_index[rank] + particles_per_rank;

    }


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



    //Force for nearby particles (neighbor cells 3x3)
    for (int i = my_start; i < my_end; i++) {
                //current grid cell of particle
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
                        if (i != j) {apply_force(parts[i], parts[j]);  // apply force
                }}
            }
        }
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


//MPI communication goes in here - 


//Communication between Rows: Use SendRecv  so that communication is more targeted 

// For Communication  - - consider ghost particles
MPI_Sendrecv(&ghost_particles_send, send_count, MPI_PARTICLE, neighbor_rank,
             0, &ghost_particles_recv, recv_count, MPI_PARTICLE, neighbor_rank, 0,
             MPI_COMM_WORLD, MPI_STATUS);




// Write this function such that at the end of it, the master (rank == 0)

//MPI_AllReduce ? or Gather?




//Simulation Time = 0.08257 seconds for 1000 particles.


// 2.284e-06 seconds for 6000000 particles. - before serial code with 2 nodes

// 2.335e-06 seconds for 6000000 particles. - before sertial code with 1 node

MPI_Finalize(); 

}


