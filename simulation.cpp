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

    double y_interval = size / num_procs;

    double starting_y = 0.0;

    for(int i = 0; i < num_procs; i++)
    {
        mpi_start_index[i] = starting_y;
        double y_cutoff = starting_y + y_interval;
        mpi_end_index[i] = y_cutoff;
        starting_y = y_cutoff;
    }
    
    //check the partioning within the mpi_start and mpi_end index vectors
    if(rank==0)
    {
        for(int i=0; i < num_procs; i++)
        {
            std::cout << "mpi_start_index[" << i << "] = " << mpi_start_index[i] <<  << "mpi_end_index[" << i << "] = " << mpi_end_index[i] << std::endl;
        }
    }


    // loop through all particles in parts
    // if the y coordinate is within my allocated y indices, add the particle to my local particles
    for(int i=0; i < num_parts; i++)
    {
        double y_coordinate = parts[i].y;
        
        if(y_coordinate > mpi_start_index[rank] && y_coordinate < mpi_end_index)
        {
            my_particles.push_back(parts[i]);
        }
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

    double top_boundary = mpi_end_index[rank] - p_cutoff;
    double bottom_boundary = mpi_start_index [rank] + p_cutoff;

    // if top boundary exceeds size, you will not have a top rank neighbor
    if(top_boundary >= size)
    {
        std::vector<particle_t> send_bottom_ghost_particles;

        for(int i = 0; i < my_particles.size(); i++)
        {
            if(my_particles[i].y < bottom_boundary)
            {
                send_bottom_ghost_particles.push_back(my_particles[i]);
            }
        }

        int bottom_send_count = send_bottom_ghost_particles.size();
        int bottom_recv_count;

        // communicate send and recv counts with my bottom neighbor
        MPI_Sendrecv(&bottom_send_count, 1, MPI_INT, rank - 1, 0,
            &bottom_recv_count, 1, MPI_INT, rank - 1, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<particles_t> recv_bottom_ghost_particles(bottom_recv_count);

        // send and receive ghost particles to my bottom neighbor
        MPI_Sendrecv(send_bottom_ghost_particles.data(), bottom_send_count, PARTICLE, rank - 1, 1,
        recv_bottom_ghost_particles.data(), bottom_recv_count, PARTICLE, rank - 1, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int i = 0; i < my_particles.size(); i++)
        {
            // apply force using the top ghost particles
            for(int j = 0; j < recv_bottom_ghost_particles.size(); j++)
            {
                apply_force(my_particles[i], recv_bottom_ghost_particles[j]);
            }
        }


    }

    // if bottom boundary is negative, you will not have a bottom rank neighbor
    else if(bottom_boundary < 0)
    {
        std::vector<particle_t> send_top_ghost_particles;
        for(int i = 0; i < my_particles.size(); i++)
        {
            if(my_particles[i].y > top_boundary)
            {
                send_top_ghost_particles.push_back(my_particles[i]);
            }
        }

        int top_send_count = send_top_ghost_particles.size();
        int top_recv_count;

        // communicate send and recv counts with my top neighbor
        MPI_Sendrecv(&top_send_count, 1, MPI_INT, rank + 1, 0,
            &top_recv_count, 1, MPI_INT, rank + 1, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<particles_t> recv_top_ghost_particles(top_recv_count);

        // send and receive ghost particles to my top neighbor
        MPI_Sendrecv(send_top_ghost_particles.data(), top_send_count, PARTICLE, rank + 1, 1,
        recv_top_ghost_particles.data(), top_recv_count, PARTICLE, rank + 1, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int i = 0; i < my_particles.size(); i++)
        {
            // apply force using the top ghost particles
            for(int j = 0; j < recv_top_ghost_particles.size(); j++)
            {
                apply_force(my_particles[i], recv_top_ghost_particles[j]);
            }
        }

    }

    // if you are a middle rank, you will have a top and bottom neighbor
    else
    {
        std::vector<particle_t> send_bottom_ghost_particles;
        std::vector<particle_t> send_top_ghost_particles;
        for(int i = 0; i < my_particles.size(); i++)
        {
            if(my_particles[i].y > top_boundary)
            {
                send_top_ghost_particles.push_back(my_particles[i]);
            }

            if(my_particles[i].y < bottom_boundary)
            {
                send_bottom_ghost_particles.push_back(my_particles[i]);
            }
        }

        int bottom_send_count = send_bottom_ghost_particles.size();
        int top_send_count = send_top_ghost_particles.size();
        int bottom_recv_count;
        int top_recv_count;

        // communication send and recv counts with my top neighbor
        MPI_Sendrecv(&top_send_count, 1, MPI_INT, rank + 1, 0,
            &top_recv_count, 1, MPI_INT, rank + 1, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // communicate send and recv counts with my bottom neighbor 
        MPI_Sendrecv(&bottom_send_count, 1, MPI_INT, rank - 1, 0,
                &bottom_recv_count, 1, MPI_INT, rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<particle_t> recv_top_ghost_particles(top_recv_count);
        std::vector<particles_t> recv_bottom_ghost_particles(bottom_recv_count);

        // send and receive ghost particles to my top neighbor
        MPI_Sendrecv(send_top_ghost_particles.data(), top_send_count, PARTICLE, rank + 1, 1,
        recv_top_ghost_particles.data(), top_recv_count, PARTICLE, rank + 1, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // send and receive ghost particles to my bottom neighbor
        MPI_Sendrecv(send_bottom_ghost_particles.data(), bottom_send_count, PARTICLE, rank - 1, 1,
        recv_bottom_ghost_particles.data(), bottom_recv_count, PARTICLE, rank - 1, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int i = 0; i < my_particles.size(); i++)
        {
            // apply force using the bottom ghost partilces
            for(int j = 0; j < recv_bottom_ghost_particles.size(); j++)
            {
                apply_force(my_particles[i], recv_bottom_ghost_particles[j]);
            }

            // apply force using the top ghost particles
            for(int j = 0; j < recv_top_ghost_particles.size(); j++)
            {
                apply_force(my_particles[i], recv_top_ghost_particles[j]);
            }

        }
        
    }

    //apply the force between my local particle pairs
    for(i = 0; i < my_particles.size(); i++)
    {
        for(int j = 0; j < my_particles.size(); j++)
        {
            if(i != j)
            {
                apply_force(my_particles[i], my_particles[j]);
            }
        }
    }

    // move all of my local particles
    for(int i = 0; i < my_particles.size(); i++)
    {
        move(my_particles[i]);
    }


    //REDISTRIBUTION

    // if top boundary exceeds size, you will not have a top rank neighbor
    if(top_boundary >= size)
    {
        std::vector<particle_t> send_bottom_particles;

        for(int i = 0; i < my_particles; i++)
        {
            if(my_particles[i].y < bottom_boundary)
            {
                send_bottom_particles.push_back(my_particles[i]);

                //remove particle from my_particles
                my_particles.erase(my_particles.begin() + i);
            }
        }

        int bottom_send_count = send_bottom_particles.size();
        int bottom_recv_count;

        // communicate send and recv counts with my bottom neighbor
        MPI_Sendrecv(&bottom_send_count, 1, MPI_INT, rank - 1, 0,
            &bottom_recv_count, 1, MPI_INT, rank - 1, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<particles_t> recv_bottom_particles(bottom_recv_count);

        // send and receive ghost particles to my bottom neighbor
        MPI_Sendrecv(send_bottom_ghost_particles.data(), bottom_send_count, PARTICLE, rank - 1, 1,
        recv_bottom_particles.data(), bottom_recv_count, PARTICLE, rank - 1, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // add the received particles to my particles
        for(int i = 0; i < recv_bottom_particles.size(); i++)
        {
            my_particles.push_back(recv_bottom_particles[i]);
        }

    }
    

    // if bottom boundary is negative, you will not have a bottom rank neighbor
    else if(bottom_boundary < 0)
    {
        std::vector<particle_t> send_top_particles;
        for(int i = 0; i < my_particles; i++)
        {
            if(my_particles[i].y > top_boundary)
            {
                send_top_particles.push_back(my_particles[i]);

                //remove particle from my_particles
                my_particles.erase(my_particles.begin() + i);
            }
        }

        int top_send_count = send_top_particles.size();
        int top_recv_count;

        // communicate send and recv counts with my top neighbor
        MPI_Sendrecv(&top_send_count, 1, MPI_INT, rank + 1, 0,
            &top_recv_count, 1, MPI_INT, rank + 1, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<particles_t> recv_top_particles(top_recv_count);

        // send and receive ghost particles to my top neighbor
        MPI_Sendrecv(send_top_particles.data(), top_send_count, PARTICLE, rank + 1, 1,
        recv_top_particles.data(), top_recv_count, PARTICLE, rank + 1, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        
        // add the received particles to my particles
        for(int i = 0; i < recv_top_particles.size(); i++)
        {
            my_particles.push_back(recv_top_particles[i]);
        }

    }


    // if you are a middle rank, you will have a top and bottom neighbor
    else
    {
        std::vector<particle_t> send_bottom_particles;
        std::vector<particle_t> send_top_particles;
        for(int i = 0; i < my_particles.size(); i++)
        {
            if(my_particles[i].y > mpi_end_index[rank])
            {
                send_top_particles.push_back(my_particles[i]);
                //remove particle from my_particles
                my_particles.erase(my_particles.begin() + i);
            }

            if(my_particles[i].y < mpi_start_index[rank])
            {
                send_bottom_particles.push_back(my_particles[i]);
                //remove particle from my_particles
                my_particles.erase(my_particles.begin() + i);
            }
        }

        int bottom_send_count = send_bottom_particles.size();
        int top_send_count = send_top_particles.size();
        int bottom_recv_count;
        int top_recv_count;

        // communication send and recv counts with my top neighbor
        MPI_Sendrecv(&top_send_count, 1, MPI_INT, rank + 1, 0,
            &top_recv_count, 1, MPI_INT, rank + 1, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // communicate send and recv counts with my bottom neighbor 
        MPI_Sendrecv(&bottom_send_count, 1, MPI_INT, rank - 1, 0,
                &bottom_recv_count, 1, MPI_INT, rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<particle_t> recv_top_particles(top_recv_count);
        std::vector<particles_t> recv_bottom_particles(bottom_recv_count);

        // send and receive ghost particles to my top neighbor
        MPI_Sendrecv(send_top_particles.data(), top_send_count, PARTICLE, rank + 1, 1,
        recv_top_particles.data(), top_recv_count, PARTICLE, rank + 1, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // send and receive ghost particles to my bottom neighbor
        MPI_Sendrecv(send_bottom_particles.data(), bottom_send_count, PARTICLE, rank - 1, 1,
        recv_bottom_particles.data(), bottom_recv_count, PARTICLE, rank - 1, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int i = 0; i < my_particles.size(); i++)
        {
            // apply force using the bottom ghost partilces
            for(int j = 0; j < recv_bottom_particles.size(); j++)
            {
                my_particles.push_back(recv_bottom_particles[i]);
            }

            // apply force using the top ghost particles
            for(int j = 0; j < recv_top_particles.size(); j++)
            {
                my_particles.push_back(recv_top_particles[i]);
            }
        }     
    }


    MPI_Barrier(MPI_COMM_WORLD);  // Sync before redistribution
}
