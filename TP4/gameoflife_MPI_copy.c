
/*
 * Conway's Game of Life
 *
 * A. Mucherino
 *
 * PPAR, TP4
 *
 */

#ifdef _WIN32
#include <windows.h> //For windows Sleep() & gethostname
#endif

#ifdef linux
#include <unistd.h> //For linux sleep()
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gameoflife.h"
#include "mpi.h"


int allReduceChange( int *my_rank) {
    int receivedValue;
    MPI_Allreduce(my_rank, &receivedValue, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return receivedValue;
}

void gatherResults(unsigned int *world1, int my_rank, int startRowId, int nbRows) {
    int numberOfCellsToSend = N * nbRows;

    unsigned int * localData = (unsigned int *) calloc(numberOfCellsToSend, sizeof(unsigned int));
    unsigned int * globalData = NULL;

    //Create buffer to receive
    if(my_rank==0){
        globalData = (unsigned int *) calloc(N * N, sizeof(unsigned int));
    }
    //Prepare part to send
    unsigned int cellToStartFrom = code(0,startRowId,0,0);
    memcpy(localData, &world1[cellToStartFrom], numberOfCellsToSend);

    MPI_Gather(&localData,numberOfCellsToSend,MPI_UNSIGNED,globalData,numberOfCellsToSend,MPI_UNSIGNED,0,MPI_COMM_WORLD);

    //update world for root
    if(my_rank==0){
        world1 = globalData;
    }
}


/**
 * Get the id of the thread managing the previous region (previous columns)
 * @param currentId, the id of the proc
 * @param nbProc, the total number of proc
 */
int getPreviousThreadId(int currentId, int nbProc){

    if(currentId == 0){
        return nbProc-1;
    }
    return currentId-1;
}

/**
 * Get the id of the thread managing the next region (next columns)
 * @param currentId, the id of the proc
 * @param nbProc, the total number of proc
 */
int getNextThreadId(int currentId, int nbProc){
    if(currentId == (nbProc-1)){
        return 0;
    }
    return currentId+1;
}


void sendTopRowAndCollect(unsigned int *world1, int startRowId, int endRowId,int my_rank,int mpi_size) {

    //SEND
    //Prepare data to send (first row)
    unsigned int * msg_to_send = (unsigned int *) calloc(N, sizeof(unsigned int));
    unsigned int cellIndexToStartFrom = code(0,startRowId,0,0);
    memcpy(msg_to_send, &world1[cellIndexToStartFrom], N); /* void *memcpy(void *dest, const void * src, size_t n) */
    //Get destination
    int previousThreadId = getPreviousThreadId(my_rank, mpi_size);
    MPI_Request request;
    //Send to previous thread (to upper rows)
    MPI_Isend(msg_to_send, N, MPI_UNSIGNED, previousThreadId, 0, MPI_COMM_WORLD, &request);

    //RECEIVE
    //Prepare to receive data
    unsigned int * msg_to_recv = (unsigned int *) calloc(N, sizeof(unsigned int));
    //Get source
    int nextThreadId = getNextThreadId(my_rank, mpi_size);
    MPI_Status status;
    //Receive from next thread (from lower rows)
    MPI_Recv(msg_to_recv, N, MPI_UNSIGNED, nextThreadId, 0, MPI_COMM_WORLD, &status);
    MPI_Wait(&request, &status);

    //Once received, update our world
    int cellIndexToUpdateFrom = code(0,endRowId,0,0);
    memcpy(&world1[cellIndexToUpdateFrom],msg_to_recv, N);
}

void sendBottomRowAndCollect(unsigned int *world1, int startRowId, int endRowId,int my_rank,int mpi_size) {
    //SEND
    //Prepare data to send (last inbound row)
    unsigned int * msg_to_send = (unsigned int *) calloc(N, sizeof(unsigned int));
    unsigned int cellIndexToStartFrom = code(0,endRowId,0,-1);
    memcpy(msg_to_send, &world1[cellIndexToStartFrom], N); /* void *memcpy(void *dest, const void * src, size_t n) */
    //Get destination
    int nextThreadId = getNextThreadId(my_rank, mpi_size);
    MPI_Request request;
    //Send to next thread (to lower rows)
    MPI_Isend(msg_to_send, N, MPI_UNSIGNED, nextThreadId, 0, MPI_COMM_WORLD, &request);

    //RECEIVE
    //Prepare to receive data
    unsigned int * msg_to_recv = (unsigned int *) calloc(N, sizeof(unsigned int));
    //Get source
    int previousThreadId = getPreviousThreadId(my_rank, mpi_size);
    MPI_Status status;
    //Receive from next thread (from upper rows)
    MPI_Recv(msg_to_recv, N, MPI_UNSIGNED, previousThreadId, 0, MPI_COMM_WORLD, &status);
    MPI_Wait(&request, &status);

    //Once received, update our world
    int cellIndexToUpdateFrom = code(0,startRowId,0,-1);
    memcpy(&world1[cellIndexToUpdateFrom],msg_to_recv, N);
}

void exchangeNeighbouringRows(unsigned int *world1, int my_rank, int mpi_size, int startRowId, int endRowId) {

    //GIVE TOP & RECEIVE BOT
    sendTopRowAndCollect(world1, startRowId, endRowId,my_rank,mpi_size);

    //GIVE BOT & RECEIVE TOP SIDE
    sendBottomRowAndCollect(world1, startRowId, endRowId,my_rank,mpi_size);

}

void printThreadStatus(int my_rank, int mpi_size) {
    char hostname[128];
    gethostname(hostname,128);
    if(my_rank == 0){
        printf("I am the master on %s\n",hostname);
    }
    else {
        printf("I am a worker on %s (rank=%d/%d)\n",hostname, my_rank,mpi_size-1);
    }
    fflush(stdout);
}


/**
 * Init world1 & world2 for each thread
 * @param my_rank, rank of thread
 * @param world1, pointer to w1 array
 * @param world2, pointer to w2 array
 */
void initWorlds(int my_rank, unsigned int *world1, unsigned int *world2) {
    // MPI : process 0 generates the initial world
    if(my_rank == 0){ //master code
        //world1 = initialize_dummy();
        //world1 = initialize_random();
        world1 = initialize_glider();
        //world1 = initialize_small_exploder();
    }
    else{
        world1 = allocate();
    }

    int worldSize = N * N ;
    //MPI : process 0 sends to all other processes the generated initial world
    MPI_Bcast(world1, worldSize, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    world2 = allocate();
}



int main(int argc, char **argv) {

    //Init random generator
    srand(time(NULL));

    //INIT VARS
    int it, change;
    unsigned int *world1 = NULL, *world2 = NULL;
    unsigned int *worldaux;
    int my_rank, mpi_size, startRowId, endRowId, nbRows; //MPI : Added

    //MPI Init
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);


    // MPI : all processes verify that N is divisible by p: if not, the execution is aborted;
    if(N%mpi_size != 0){
        if(my_rank==0){
            printf("The program is supposed to run with N%%nbProc==0.Please try with a different value.\n");
            fflush(stdout);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    //Print status of thread
    printThreadStatus(my_rank, mpi_size);

    // CREATE WORLD
    initWorlds(my_rank, world1, world2);

    //MPI : • every process computes the first and last row index of its world region;
    nbRows = N % mpi_size;
    startRowId = nbRows * my_rank; //(incl.)
    endRowId = nbRows * (my_rank + 1); //(excl.)

    //Loop for itMax generations
    it = 0;
    change = 1;
    while (change && it < itMax) {
        //MPI : every process invokes newgeneration() with its first and last row index
        change = newGeneration(world1, world2, startRowId, endRowId);
        //MPI : every process invokes newgeneration() with its first and last row index
        worldaux = world1;
        world1 = world2;
        world2 = worldaux;

        //MPI : the processes exchange their neighbouring rows, necessary for computing the next generation
        //(communication type: one-to-one);
        exchangeNeighbouringRows(world1, my_rank, mpi_size, startRowId, endRowId);

        //MPI :  process 0 collects the results obtained by the other processes
        gatherResults(world1, my_rank, startRowId, nbRows);

        if(my_rank==0){
            print(world1);
            fflush(stdout);
        }

        //MPI : check if continue
        // Each MPI process sends its rank to reduction, root MPI process collects the result
        /*Op types
         * https://www.mcs.anl.gov/research/projects/mpi/mpi-standard/mpi-report-1.1/node78.htm#Node78
         * Here, if one is 1, we'll propagate 1
         */
        change = allReduceChange(&my_rank);

        it++;
    }

    // ending
    free(world1);
    free(world2);

    MPI_Finalize();
    return 0;
}



