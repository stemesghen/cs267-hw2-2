
#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
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

    std::cout << "Applying force between particles:"
          << " P1(id=" << particle.id << ", x=" << particle.x << ", y=" << particle.y << ")"
          << " P2(id=" << neighbor.id << ", x=" << neighbor.x << ", y=" << neighbor.y << ")"
          << " Distance=" << sqrt(r2) 
          << " r2=" << r2
          << std::endl;

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


    mpi_start_index.resize(num_procs);
    mpi_end_index.resize(num_procs);
    

    int particles_per_rank = num_parts / num_procs;
    int remaining_particles = num_parts % num_procs;


    int start_rank_index, end_rank_index;
    if (rank < remaining_particles) {
                start_rank_index = rank * (particles_per_rank + 1);
                end_rank_index = start_rank_index + particles_per_rank + 1;
    } 

    else {
        start_rank_index = rank * particles_per_rank + remaining_particles;
        end_rank_index = start_rank_index + particles_per_rank;

    }


    // Store these indices for each rank
    mpi_start_index[rank] = start_rank_index;
    mpi_end_index[rank] = end_rank_index;

MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks sync before debugging

    // Test Partitioning Across Ranks
    std::cout << "Rank " << rank << " Start Index: " << mpi_start_index[rank] 
              << " End Index: " << mpi_end_index[rank] << std::endl;
    
    // Run Single-Processor Validation Before MPI
    if (num_procs == 1) {
        std::cout << "Running single-processor mode to verify correctness." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD); // Sync before proceeding
    
}




void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs)

{
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



// In this, it is a grid that is used for spatial partitioning within the rows
    // grid_size = size / cutoff
    const double p_cutoff = 0.01;
    const int grid_size = ceil(size / p_cutoff); // # of grid cells per dim
    // each grid cell includes a list of particle indices 
    std::vector<std::vector<int>> grid(grid_size * grid_size);

    // loop to assign each particle to a grid cell
    for (int i = 0; i < num_parts; i++) 
    {
        // mapping particle position
        int cell_x = floor(parts[i].x / p_cutoff);
        int cell_y = floor(parts[i].y / p_cutoff);
        grid[cell_x + cell_y * grid_size].push_back(i);
    }


std::vector<particle_t> top_ghost_particles, bottom_ghost_particles;
std::vector<particle_t> recv_top_ghosts, recv_bottom_ghosts;

std::cout << "Rank " << rank << " Sending " << top_ghost_particles.size()
          << " to Rank " << (rank + 1) << " and receiving " 
          << recv_bottom_ghosts.size() << " in return." << std::endl;


//Considering Ghost particles (identifying the location of the ghost particles)
//                - dynamic array or list or another data structure to consider the particles within the cutoff range




for (int i = mpi_start_index[rank]; i < mpi_end_index[rank]; i++) {
    if (parts[i].y >= (mpi_end_index[rank] - cutoff)) {  // Top boundary particles
        top_ghost_particles.push_back(parts[i]);
    }
    if (parts[i].y <= (mpi_start_index[rank] + cutoff)) {  // Bottom boundary particles
        bottom_ghost_particles.push_back(parts[i]);
    } 
}
std::cout << "Rank " << rank << " Range: Y_start=" << mpi_start_index[rank] 
          << " Y_end=" << mpi_end_index[rank] << std::endl;

// Communication -  ONLY for ghost particles before force calculation happens: 
             //   - so use point to point communication (Send Recv with that neighbor)




if (rank < num_procs - 1) { // Exchange with the rank above
    int send_count = top_ghost_particles.size();
    int recv_count;

    MPI_Sendrecv(&send_count, 1, MPI_INT, rank + 1, 0,  // Send count to next rank
                 &recv_count, 1, MPI_INT, rank + 1, 0,  // Receive count from next rank
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Print particle count exchange

    recv_bottom_ghosts.resize(recv_count);

    std::cout << "Rank " << rank << " sending " << send_count 
              << " particles to Rank " << (rank + 1)
              << " and expecting " << recv_count << " in return." << std::endl;

        // Actual particle data exchange

    MPI_Sendrecv(top_ghost_particles.data(), send_count, PARTICLE, rank + 1, 1,  // Send top ghosts to next rank
                 recv_bottom_ghosts.data(), recv_count, PARTICLE, rank + 1, 1,  // Receive bottom ghosts from next rank
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

 // Print confirmation after actual particle data is received
    std::cout << "Rank " << rank << " received " << recv_count 
              << " particles from Rank " << rank + 1 << "." << std::endl;


    }


if (rank > 0) { // Exchange with the rank below
    int send_count = bottom_ghost_particles.size();
    int recv_count;

    MPI_Sendrecv(&send_count, 1, MPI_INT, rank - 1, 0,  // Send count to previous rank
                 &recv_count, 1, MPI_INT, rank - 1, 0,  // Receive count from previous rank
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
 
    recv_top_ghosts.resize(recv_count);

std::cout << "Rank " << rank << " sending " << send_count 
              << " particles to Rank " << (rank - 1)
              << " and expecting " << recv_count << " in return." << std::endl;

    MPI_Sendrecv(bottom_ghost_particles.data(), send_count, PARTICLE, rank - 1, 1,  // Send bottom ghosts to previous rank
                 recv_top_ghosts.data(), recv_count, PARTICLE, rank - 1, 1,  // Receive top ghosts from previous rank
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                  // Print confirmation after actual particle data is received
    std::cout << "Rank " << rank << " received " << recv_count 
              << " particles from Rank " << rank - 1 << "." << std::endl;
    }
                 

//Print final ghost particle exchange summary per rank
std::cout << "Rank " << rank << " Final Ghost Exchange Summary:"
          << "\n\tSending " << top_ghost_particles.size() << " particles UP."
          << "\n\tReceiving " << recv_bottom_ghosts.size() << " particles from BELOW."
          << "\n\tSending " << bottom_ghost_particles.size() << " particles DOWN."
          << "\n\tReceiving " << recv_top_ghosts.size() << " particles from ABOVE."
          << std::endl;
/*
    Runs simulation per process 
    May need to redistribute the particles within this function 
    Because the next row processor has to have the correct set of particles to sumulate
    Some particles may go across the different processors after each step. 
*/


std::cout << "Rank " << rank << " received " << recv_top_ghosts.size() 
          << " from top, " << recv_bottom_ghosts.size() << " from bottom." << std::endl;


// Reset acceleration for all local particles
for (int i = 0; i < num_parts; ++i) {
    parts[i].ax = parts[i].ay = 0;
}

// Loop through each local particle
for (int i = 0; i < num_parts; ++i) {
    int cell_x = (int)(floor(parts[i].x / cutoff));
    int cell_y = (int)(floor(parts[i].y / cutoff));

    // Ensure cell indices are within bounds
    if (cell_x < 0 || cell_x >= num_cells_x || cell_y < 0 || cell_y >= num_cells_y)
        continue;

    // 3×3 neighborhood search
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            int neighbor_x = cell_x + x;
            int neighbor_y = cell_y + y;

            // Check bounds
            if (neighbor_x >= 0 && neighbor_x < num_cells_x &&
                neighbor_y >= 0 && neighbor_y < num_cells_y) {
                
                int neighbor_idx = neighbor_x + neighbor_y * num_cells_x;

                for (int j : grid[neighbor_idx]) {
                    if (&parts[i] != &parts[j]) {  // Avoid self-interactions
                        apply_force(parts[i], parts[j]);
                    }
                }
            }
        }
    }
}

// Handle ghost particle force interactions safely
for (particle_t& ghost : recv_top_ghosts) {
    if (ghost.y >= domain_height - cutoff && ghost.y <= domain_height) {
        for (int i = 0; i < num_parts; ++i) {
            apply_force(parts[i], ghost);
        }
    }
}

for (particle_t& ghost : recv_bottom_ghosts) {
    if (ghost.y >= 0 && ghost.y < cutoff) {
        for (int i = 0; i < num_parts; ++i) {
            apply_force(parts[i], ghost);
        }
    }
}

// Debugging: Check particle counts per cell
for (int c = 0; c < num_cells_x * num_cells_y; c++) {
    std::cout << "Cell " << c << " has " << grid[c].size() << " particles.\n";
}

// Debugging: Confirm forces are being applied
for (int i = 0; i < num_parts; ++i) {
    std::cout << "Particle " << i << " acceleration before: " << parts[i].ax << ", " << parts[i].ay << "\n";
}


/*

// Compute Forces: changed to only use nearby particles
// Compute Forces: changed to only use nearby particles
for (int i = 0; i < num_parts; ++i) 
{
    std::cout << "Rank " << rank << " Particle " << i 
              << " Initial ax=" << parts[i].ax 
              << " ay=" << parts[i].ay << std::endl;
              
    // Reset acceleration for each particle
    parts[i].ax = parts[i].ay = 0;
    
    // Compute cell position
    int cell_x = floor(parts[i].x / cutoff);
    int cell_y = floor(parts[i].y / cutoff);

    // 3x3 neighborhood search
    for (int x = -1; x <= 1; x++) 
    {
        for (int y = -1; y <= 1; y++) 
        {
            int neighbor_x = cell_x + x;
            int neighbor_y = cell_y + y;
            
            // Bounds check
            if (neighbor_x >= 0 && neighbor_x < grid_size &&
                neighbor_y >= 0 && neighbor_y < grid_size) 
            {
                // Get index in grid
                int neighbor_idx = neighbor_x + neighbor_y * grid_size;

                std::cout << "Rank " << rank << " Particle " << i << " in cell (" 
                          << cell_x << ", " << cell_y << ") checking neighbor cell (" 
                          << neighbor_x << ", " << neighbor_y << ") with " 
                          << grid[neighbor_idx].size() << " particles." << std::endl;

                // Loop through particles in neighboring cell
                for (int j : grid[neighbor_idx]) 
                {
                    if (i != j) apply_force(parts[i], parts[j]);
                }
            }
        }
    }

    std::cout << "Rank " << rank << " Particle " << i 
              << " Final ax=" << parts[i].ax 
              << " ay=" << parts[i].ay << std::endl;
}


*/

// if (rank == 0) {
//     particle_t p1 = {1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
//     particle_t p2 = {2, 0.5 * cutoff, 0.5 * cutoff, 0.0, 0.0, 0.0, 0.0};  // Ensure it's within range
    
//     std::cout << "Testing apply_force: Initial ax=" << p1.ax << " ay=" << p1.ay << std::endl;
    
//     apply_force(p1, p2);
    
//     std::cout << "After apply_force: ax=" << p1.ax << " ay=" << p1.ay << std::endl;
    
//     if (p1.ax == 0 && p1.ay == 0) {
//         std::cout << "Force application failed! Check calculations!" << std::endl;
//     } else {
//         std::cout << " Force applied successful!" << std::endl;
//     }
// }


        MPI_Barrier(MPI_COMM_WORLD);  // Sync before movement

    // Move Particles
    for (int i = 0; i < num_parts; ++i) 
    {
        move(parts[i], size);
    }
        // Test Movement
    if (rank == 0) {
        particle_t p = {3, size - 0.001, size - 0.001, -0.1, -0.1, 0.0, 0.0};
        move(p, size);
        std::cout << "Rank " << rank << " Test Wall Bounce: x=" << p.x << " y=" << p.y 
                  << " vx=" << p.vx << " vy=" << p.vy << std::endl;


    }
                      MPI_Barrier(MPI_COMM_WORLD);  // Sync before redistribution


    //Afte movement - Redistribute particles so that the next rank 
    //using send receive from the rank above so that the rank below which will be the current rank, can have all the infrmation
// After Movement: Redistribute Particles Across Ranks

 

    // After Movement: Redistribute Particles
    std::vector<particle_t> send_to_next, send_to_prev, new_local_particles;
for (int i = 0; i < num_parts; i++) {
    if (parts[i].y > mpi_end_index[rank] && rank < num_procs - 1) {  
        send_to_next.push_back(parts[i]);
    } 
    else if (parts[i].y < mpi_start_index[rank] && rank > 0) {  
        send_to_prev.push_back(parts[i]);
    } 
    else {
        new_local_particles.push_back(parts[i]);  
    }
}


    // Exchange Moved Particles
    std::vector<particle_t> recv_from_next, recv_from_prev;


    int send_count_next = send_to_next.size();
    int send_count_prev = send_to_prev.size();
    int recv_count_next = 0, recv_count_prev = 0;

if (rank < num_procs - 1) {
    MPI_Sendrecv(&send_count_next, 1, MPI_INT, rank + 1, 0,
                 &recv_count_prev, 1, MPI_INT, rank + 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
} else {
    recv_count_prev = 0;  // Ensure no waiting
}

if (rank > 0) {
    MPI_Sendrecv(&send_count_prev, 1, MPI_INT, rank - 1, 0,
                 &recv_count_next, 1, MPI_INT, rank - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
} else {
    recv_count_next = 0;  // Ensure no waiting
}

//  Debug Output for Correct Particle Counts
std::cout << "Rank " << rank << " expecting " << recv_count_next 
          << " from above and " << recv_count_prev << " from below." << std::endl;


// Resize buffers 
if (recv_count_next > 0) recv_from_next.resize(recv_count_next);
if (recv_count_prev > 0) recv_from_prev.resize(recv_count_prev);

// Send/Receive only if there's data
if (rank < num_procs - 1 && recv_count_next > 0) {
//  Resize buffers before receiving
recv_from_next.resize(recv_count_next);
recv_from_prev.resize(recv_count_prev);

// Correct Order for MPI Send/Recv
if (rank < num_procs - 1 && recv_count_next > 0) {
    MPI_Recv(recv_from_next.data(), recv_count_next, PARTICLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
if (rank < num_procs - 1 && send_count_next > 0) {
    MPI_Send(send_to_next.data(), send_count_next, PARTICLE, rank + 1, 1, MPI_COMM_WORLD);
}

if (rank > 0 && recv_count_prev > 0) {
    MPI_Recv(recv_from_prev.data(), recv_count_prev, PARTICLE, rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
if (rank > 0 && send_count_prev > 0) {
    MPI_Send(send_to_prev.data(), send_count_prev, PARTICLE, rank - 1, 2, MPI_COMM_WORLD);
}

//  Print Debug Information
std::cout << "Rank " << rank << " received " << recv_from_prev.size() << " from below, " 
          << recv_from_next.size() << " from above." << std::endl;

}



// Synchronize all processes to prevent race conditions
//  Append received particles correctly
new_local_particles.insert(new_local_particles.end(), recv_from_next.begin(), recv_from_next.end());
new_local_particles.insert(new_local_particles.end(), recv_from_prev.begin(), recv_from_prev.end());

MPI_Barrier(MPI_COMM_WORLD);


// Debug Total Particle Count
int total_particles = 0;
MPI_Allreduce(&num_parts, &total_particles, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
std::cout << "Rank " << rank << " now has " << num_parts << " particles. Total: " << total_particles << std::endl;

// std::cout << "Rank " << rank << " Checking Redistribution:"
//           << " local=" << new_local_particles.size()
//           << " send_up=" << send_to_next.size()
//           << " send_down=" << send_to_prev.size() << std::endl;

 //Print Redistribution Debug Info
    std::cout << "Rank " << rank << " Checking Redistribution:"
              << " local=" << num_parts 
              << " send_up=" << send_to_next.size()
              << " send_down=" << send_to_prev.size() << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);  // Sync before next iteration
}
// //Debug Final Redistribution Summary
// std::cout << "Rank " << rank << " Final Redistribution:"
//           << "\n\tSent to Next: " << send_to_next.size()
//           << "\n\tSent to Prev: " << send_to_prev.size()
//           << "\n\tReceived from Next: " << recv_from_next.size()
//           << "\n\tReceived from Prev: " << recv_from_prev.size()
//           << std::endl;




void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id. 


/* 

Gather the final posiitons of the particles and save the data (All to 1) 

    - Gather: gathers buffers from all process in a communicator
    - Gatherv: can gather different amounts of data from ALL processors. 
        - Bcast: sends the same 

Test on a single node first

When you send the process recieve has to also have the receive


*/

    std::vector<int> send_counts(num_procs);
    MPI_Gather(&num_parts, 1, MPI_INT, send_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements(num_procs);
    int total_particles = 0;
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            displacements[i] = total_particles;
            total_particles += send_counts[i];
        }
    }

    std::vector<particle_t> all_particles;
    if (rank == 0) all_particles.resize(total_particles);

    MPI_Gatherv(parts, num_parts, PARTICLE,
                all_particles.data(), send_counts.data(), displacements.data(), PARTICLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });

        std::cout << "Rank " << rank << " Final Gather: Collected " 
                  << all_particles.size() << " particles." << std::endl;
    }

}





