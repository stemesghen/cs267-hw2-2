void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Compute Forces


    // establish the start and end index of my_rank
    int my_start_index = mpi_start_index[rank];
    int my_end_index = mpi_end_index[rank];

    std::vector<particle_t> my_particles;
    // create array that corresponds to current rank
    for(int i = my_start_index; i < my_end_index; i++)
    {
        my_particles.push_back(parts[i]);
    }

    // establish ring communication parameetrs
    int send_count = my_particles.size();
    std::vector<particle_t> send_vec = my_particles;
    //std::cout << "my rank: " << rank << " send_vec size: " << send_vec.size() << std::endl;

    int recv_rank = rank -1;
    if(recv_rank < 0) recv_rank = num_procs - 1;

    //std::cout << "my rank: " << rank << " send rank: " << (rank + 1) % num_procs << " recv_rank: " << recv_rank << std::endl;

        //begin ring communcation
    for (int istep = 0; istep < num_procs; istep++)
    {
        int recv_origin_rank = num_procs - 1 - istep;

        if (recv_origin_rank < 0) recv_origin_rank += num_procs;

        int recv_count = (mpi_end_index[recv_origin_rank] - mpi_start_index[recv_origin_rank]);

        //std::cout << "my_rank: " << rank << " recv_origin_rank: " << recv_origin_rank << " recv_count: " << recv_count << std::endl;

        std::vector<particle_t> recv_vec(recv_count);

       //std::cout << "my_rank: " << rank << " recv_origin_rank: " << recv_origin_rank << " recv_count: " << recv_count << " recv_vec size: " << recv_vec.size()  << " send count: " << send_count << std::endl;

         if(rank == recv_origin_rank)
        {
            recv_vec = my_particles;
        }
        else
        {
        //std::cout << "my_rank: " << rank << " recv_origin_rank: " << recv_origin_rank << " recv_count: " << recv_count << " recv_vec size: " << recv_vec.size()  << " send count: " << send_count << std::endl;

        MPI_Sendrecv(send_vec.data(), send_count, PARTICLE, (rank + 1) % num_procs, 0,
                    recv_vec.data(), recv_count, PARTICLE, recv_rank, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

       // update send variables for next iteration
        send_vec = recv_vec;
        send_count = recv_count;
        int recv_start_index = mpi_start_index[recv_origin_rank];
        int recv_end_index = mpi_end_index[recv_origin_rank];


            // Compute Forces
        for (int i = my_start_index; i < my_end_index; ++i)
        {
            parts[i].ax = parts[i].ay = 0;
            for (int j = recv_start_index; j < recv_end_index; ++j)
            {
                apply_force(parts[i], parts[j]);
            }
        }
    }
     // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}
