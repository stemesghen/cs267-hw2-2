#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstddef> 


// Apply force between two particles
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

// Move the particle using Velocity Verlet integration
void move(particle_t& p, double size) {
    p.x += p.vx * dt + 0.5 * p.ax * dt * dt;
    p.y += p.vy * dt + 0.5 * p.ay * dt * dt;

    p.vx += 0.5 * p.ax * dt;
    p.vy += 0.5 * p.ay * dt;

    while (p.x < 0 || p.x > size) {
        p.x = (p.x < 0) ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = (p.y < 0) ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

}