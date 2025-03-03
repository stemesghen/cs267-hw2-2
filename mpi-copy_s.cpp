#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
// Put any static global variables here that you will use throughout the simulation.



// Apply the force from a neighbor to a particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;  // Ignore if beyond cutoff distance

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    double coef = (1 - cutoff / r) / r2 / mass;  // Short-range repulsion
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


    int grid_size = ceil(size / cutoff);  //grid size

    std::vector<std::vector<int>> grid(grid_size * grid_size);
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;

        int grid_cell_x = floor(parts[i].x / cutoff);         //current grid cell of particle

        int grid_cell_y = floor(parts[i].y / cutoff);

        int grid_index = grid_cell_x + grid_cell_y * grid_size;  // create 1d from 2d
        grid[grid_index].push_back(i); 

    }



    //Force for nearby particles (neighbor cells 3x3)
    for (int i = 0; i < num_parts; i++) {
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
    for (int i = 0; i < num_parts; ++i) 
    {
        move(parts[i], size);
    }
}



void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id. 


//MPI communication goes in here - 

//Simulation Time = 0.08257 seconds for 1000 particles.


// 2.284e-06 seconds for 6000000 particles. - before serial code with 2 nodes

// 2.335e-06 seconds for 6000000 particles. - before sertial code with 1 node

}
