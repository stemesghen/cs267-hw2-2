




// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {

if (particle.id == neighbor.id) return; // Avoid self-interaction

    std::cout << "Applying force: P" << particle.id << " → P" << neighbor.id << std::endl;


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

    // std::cout << "Force applied between P" << particle.id << " and P" << neighbor.id << std::endl;


}








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

                // Loop through particles in neighboring cell
                for (int j : grid[neighbor_idx]) 
                {
                    if (i != j) 
                    {  
                        // Compute squared distance
                        double dx = parts[j].x - parts[i].x;
                        double dy = parts[j].y - parts[i].y;
                        double r2 = dx * dx + dy * dy;

                        // 🔹 FIX: Ensure particles exactly at cutoff distance are included
                        if (r2 > cutoff * cutoff + 1e-12)  
                            continue;  // Skip if outside interaction range

                        // 🔹 Debugging: Check if we are applying force correctly
                        std::cout << "[Rank " << rank << "] Checking force application: "
                                  << " P" << parts[i].id << " → P" << parts[j].id 
                                  << " | Distance^2: " << r2
                                  << " | Cutoff^2: " << cutoff * cutoff
                                  << std::endl;

                        // Apply force
                        apply_force(parts[i], parts[j]);

                        // 🔹 Debugging: Confirm force was applied
                        std::cout << "[Rank " << rank << "] Force applied: "
                                  << "P" << parts[i].id << " (ax=" << parts[i].ax 
                                  << ", ay=" << parts[i].ay << ") due to P" << parts[j].id 
                                  << std::endl;
                    }
                }
            }
        }
    }

    std::cout << "Rank " << rank << " Particle " << i 
              << " Final ax=" << parts[i].ax 
              << " ay=" << parts[i].ay << std::endl;
}
