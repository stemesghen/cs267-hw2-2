MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
    }

        // If using MPI_IN_PLACE, load local particles into the correct location
    if (rank == 0) {
        // Nothing needs to be done, the data is already in the correct location
    } else {
        // Non-root ranks copy their data to the correct position in parts array
        for (int i = 0; i < num_parts; i++) {
            parts[i + displs[rank]] = parts[i];
        }
    }

    /*
    int start_index = displs[rank];

    for(int i = 0; i < num_parts; i++)
    {   
            parts[i + start_index] = parts[i];
    }
    
    // Temporarily store rank 0's local data
    std::vector<PARTICLE> temp_parts;
    if (rank == 0)
    {
        temp_parts = parts;  
    }

    // Resize the parts array on rank 0
    if (rank == 0)
    {
        parts.resize(total_parts);
    }

    // After resizing, copy the local data of rank 0 back into parts
    if (rank == 0) 
    {
        std::copy(temp_parts.begin(), temp_parts.end(), parts.begin());
    }
    */

   // MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : parts, num_parts, PARTICLE, particles.data(),
         // recv_counts.data(), displs.data(), PARTICLE, 0, MPI_COMM_WORLD);
   MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : parts, num_parts, PARTICLE, all_parts.data(), recv_counts.data(), displs.data(), PARTICLE, 0, MPI_COMM_WORLD);
