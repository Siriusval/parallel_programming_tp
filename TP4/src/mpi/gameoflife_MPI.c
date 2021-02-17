/**
 * @author Valou
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
#include "../lib/inc/gameoflife.h"
#include "gameoflife_MPI.h"
#include "mpi.h"

int allReduceChange(int *change) {
    int reduction_result = 0;

    MPI_Allreduce(change, &reduction_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return reduction_result;
}

void gatherResults(unsigned int **world1, int my_rank, int mpi_size, int startRowId, int nbRows) {
    int numberOfCellsToSend = N * nbRows;

    unsigned int * localData = (unsigned int *) calloc(numberOfCellsToSend, sizeof(unsigned int));
    unsigned int * globalData = NULL;

    //Create buffer to receive
    if(my_rank==0){
        globalData = (unsigned int *) calloc(mpi_size*numberOfCellsToSend, sizeof(unsigned int));
    }
    //Prepare part to send
    unsigned int cellToStartFrom = code(0,startRowId,0,0);
    unsigned int * worldData = *world1;

    memcpy(localData, &worldData[cellToStartFrom], numberOfCellsToSend*sizeof(unsigned int));

    MPI_Gather(localData,numberOfCellsToSend,MPI_UNSIGNED,globalData,numberOfCellsToSend,MPI_UNSIGNED,0,MPI_COMM_WORLD);

    //update world for root
    if(my_rank==0){
        *world1 = globalData;
    }

}

int getPreviousThreadId(int currentId, int nbProc) {

    if(currentId == 0){
        return nbProc-1;
    }
    return currentId-1;
}

int getNextThreadId(int currentId, int nbProc) {
    if(currentId == (nbProc-1)){
        return 0;
    }
    return currentId+1;
}



void sendTopRowAndCollect(unsigned int *world1,int my_rank, int mpi_size,int startRowId, int endRowId) {

    //SEND
    //Prepare data to send (first row)
    unsigned int * msg_to_send = (unsigned int *) calloc(N, sizeof(unsigned int));
    unsigned int cellIndexToStartFrom = code(0,startRowId,0,0);

    memcpy(msg_to_send, &world1[cellIndexToStartFrom], N*sizeof(unsigned int)); /* void *memcpy(void *dest, const void * src, size_t n) */

    //Get destination
    int previousThreadId = getPreviousThreadId(my_rank, mpi_size);
    MPI_Request request;
    //Send to previous thread (to upper rows)
    MPI_Isend(msg_to_send, N, MPI_UNSIGNED, previousThreadId, 0, MPI_COMM_WORLD, &request);
    printf("proc %d sent top row to %d.\t",my_rank,previousThreadId);
    //for(int i =0;i<N;i++) printf("%d",msg_to_send[i]);printf("\n");

    //RECEIVE
    //Prepare to receive data
    unsigned int * msg_to_recv = (unsigned int *) calloc(N, sizeof(unsigned int));
    //Get source
    int nextThreadId = getNextThreadId(my_rank, mpi_size);
    MPI_Status status;
    //Receive from next thread (from lower rows)
    MPI_Recv(msg_to_recv, N, MPI_UNSIGNED, nextThreadId, 0, MPI_COMM_WORLD, &status);
    MPI_Wait(&request, &status);
    printf("proc %d received bottom row from %d\n",my_rank,nextThreadId);
    //for(int i =0;i<N;i++) printf("%d",msg_to_recv[i]);printf("\n");
    fflush(stdout);

    //Once received, update our world
    int cellIndexToUpdateFrom = code(0,endRowId,0,0);
    memcpy(&world1[cellIndexToUpdateFrom],msg_to_recv, N*sizeof(unsigned int));
}


void sendBottomRowAndCollect(unsigned int *world1, int my_rank, int mpi_size,int startRowId, int endRowId) {
    //SEND
    //Prepare data to send (last inbound row)
    unsigned int * msg_to_send = (unsigned int *) calloc(N, sizeof(unsigned int));
    unsigned int cellIndexToStartFrom = code(0,endRowId,0,-1);
    memcpy(msg_to_send, &world1[cellIndexToStartFrom], N*sizeof(unsigned int)); /* void *memcpy(void *dest, const void * src, size_t n) */
    //Get destination
    int nextThreadId = getNextThreadId(my_rank, mpi_size);
    MPI_Request request;
    //Send to next thread (to lower rows)
    MPI_Isend(msg_to_send, N, MPI_UNSIGNED, nextThreadId, 0, MPI_COMM_WORLD, &request);
    printf("proc %d sent bottom row to %d.\t",my_rank,nextThreadId);

    //RECEIVE
    //Prepare to receive data
    unsigned int * msg_to_recv = (unsigned int *) calloc(N, sizeof(unsigned int));
    //Get source
    int previousThreadId = getPreviousThreadId(my_rank, mpi_size);
    MPI_Status status;
    //Receive from next thread (from upper rows)
    MPI_Recv(msg_to_recv, N, MPI_UNSIGNED, previousThreadId, 0, MPI_COMM_WORLD, &status);
    MPI_Wait(&request, &status);
    printf("proc %d received top row from %d\n",my_rank,previousThreadId);
    //for(int i =0;i<N;i++) printf("%d",msg_to_recv[i]);printf("\n");
    fflush(stdout);

    //Once received, update our world
    int cellIndexToUpdateFrom = code(0,startRowId,0,-1);
    memcpy(&world1[cellIndexToUpdateFrom],msg_to_recv, N*sizeof(unsigned int));

}


void exchangeNeighbouringRows(unsigned int *world1, int my_rank, int mpi_size, int startRowId, int endRowId) {

    //GIVE TOP & RECEIVE BOT
    sendTopRowAndCollect(world1,my_rank,mpi_size, startRowId, endRowId);

    //GIVE BOT & RECEIVE TOP SIDE
    sendBottomRowAndCollect(world1,my_rank,mpi_size, startRowId, endRowId);
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

void initWorlds(int my_rank, unsigned int **world1, unsigned int **world2,int gameMode) {
    // MPI : process 0 generates the initial world
    if(my_rank == 0){ //master code
        *world1 = createWorld(gameMode);
    }
    else{
        *world1 = allocate();
    }

    int worldSize = N * N ;
    //MPI : process 0 sends to all other processes the generated initial world
    MPI_Bcast(*world1, worldSize, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    //Console log
    if(my_rank==0){
        printf("World broadcasted\n");
    }
    else {
        printf("Proc %d : World received\n",my_rank);
    }
    fflush(stdout);

    *world2 = allocate();
}


int main(int argc, char **argv) {

    int my_rank, mpi_size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);

    srand(time(NULL));

    //INIT VARS
    int it, change;
    unsigned int *world1 = NULL, *world2 = NULL;
    unsigned int *worldAux;
    int  startRowId, endRowId, nbRows; //MPI : Added

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
    MPI_Barrier(MPI_COMM_WORLD);

    int input = 0;
    // CREATE WORLD
    //ask game mode
    if(my_rank==0){
        input = askGameMode();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /*MPI : process 0 generates the initial world
     *      process 0 sends to all other processes the generated initial world
     *      (communication type:one-to-all);
     */
    initWorlds(my_rank, &world1, &world2,input);

    //MPI : process 0 prints the initial world on the screen
    if(my_rank==0){
        print(world1);
        fflush(stdout);
    }
    //MPI : • every process computes the first and last row index of its world region;
    nbRows = N / mpi_size;
    startRowId = nbRows * my_rank; //(incl.)
    endRowId = nbRows * (my_rank + 1); //(excl.)

    //Loop for itMax generations
    it = 0;
    change = 1;
    while (change && it < itMax) {
        //MPI : every process invokes newGeneration() with its first and last row index
        change = newGeneration(world1, world2, startRowId, endRowId);

        worldAux = world1;
        world1 = world2;
        world2 = worldAux;

        //MPI : the processes exchange their neighbouring rows, necessary for computing the next generation
        //(communication type: one-to-one);

        exchangeNeighbouringRows(world1, my_rank, mpi_size, startRowId, endRowId);

        //MPI :  process 0 collects the results obtained by the other processes
        gatherResults(&world1, my_rank,mpi_size, startRowId, nbRows);

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
        change = allReduceChange(&change);

        it++;
    }

    // ending
    free(world1);
    free(world2);


    MPI_Finalize();
    return 0;
}





