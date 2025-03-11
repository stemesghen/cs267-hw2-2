#include "common.h"
#include <cmath>
#include <vector>

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


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
}

void simulate_one_step(particle_t* parts, int num_parts, double size) 
{
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

    // Compute Forces: changed to only use nearby particles
    for (int i = 0; i < num_parts; ++i) 
    {
        // acceleration resets for each particle
        parts[i].ax = parts[i].ay = 0;
        // x position to account for movement
        int cell_x = floor(parts[i].x / cutoff);
        // y position to account for movement
        int cell_y = floor(parts[i].y / cutoff);

        // 3x3 neighborhood search
        // loop through neighbor grid cells
        for (int x = -1; x <= 1; x++) 
        {
            for (int y = -1; y <= 1; y++) 
            {
                // neighbors position
                int neighbor_x = cell_x + x;
                int neighbor_y = cell_y + y;
                
                //  bounds check
                if (neighbor_x >= 0 && neighbor_x < grid_size &&
                    neighbor_y >= 0 && neighbor_y < grid_size) 
                {
                    // reshape: (x,y) -> int
                    int neighbor_idx = neighbor_x + neighbor_y * grid_size;
                    
                    // loop through particles in neighbor cell
                    for (int j : grid[neighbor_idx]) 
                    {
                        // compute force if different particle
                        if (i != j) apply_force(parts[i], parts[j]);
                    }
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
