/**
 * @author Valou
 */
#ifndef TP4_GAMEOFLIFE_MPI_H
#define TP4_GAMEOFLIFE_MPI_H

/**
 * Compute if the world has changed in any thread
 * @param change, the change to send for reduce
 * @return the global change value
 */
int allReduceChange( int *change);

/**
 * Gather the parts of world1 of each thread in master node
 * @param world1, the place where data of the world is gathered
 * @param my_rank, rank of the current thread
 * @param mpi_size, total number of threads
 * @param startRowId, first row of region for thread
 * @param nbRows, number of rows in region
 */
void gatherResults(unsigned int **world1, int my_rank,int mpi_size, int startRowId, int nbRows);

/**
 * Get the id of the thread managing the previous region (previous columns)
 * @param currentId, the id of the proc
 * @param nbProc, the total number of proc
 * @return the id of the previous thread
 */
int getPreviousThreadId(int currentId, int nbProc);

/**
 * Get the id of the thread managing the next region (next columns)
 * @param currentId, the id of the proc
 * @param nbProc, the total number of proc
 * @return the id of the previous thread
 */
int getNextThreadId(int currentId, int nbProc);

/**
 * Allow thread to send the topmost row of his region to n-1 thread
 * Collect the row sent to him by thread n+1, and update world
 * @param world1, the world containing the row & world to update
 * @param startRowId, starting row of the region
 * @param endRowId, ending row of the region
 * @param my_rank, rank of the thread
 * @param mpi_size, total number of thread
 */
void sendTopRowAndCollect(unsigned int *world1,int my_rank,int mpi_size, int startRowId, int endRowId);

/**
 * Allow thread to send the bottom row of his region to n+1 thread
 * Collect the row sent to him by thread n-1, and update world
 * @param world1, the world containing the row & world to update
 * @param startRowId, starting row of the region
 * @param endRowId, ending row of the region
 * @param my_rank, rank of the thread
 * @param mpi_size, total number of thread
 */
void sendBottomRowAndCollect(unsigned int *world1,int my_rank,int mpi_size, int startRowId, int endRowId);

/**
 * Allow thread to exchange rows with neighbours for world update
 * @param world1
 * @param my_rank
 * @param mpi_size
 * @param startRowId
 * @param endRowId
 */
void exchangeNeighbouringRows(unsigned int *world1, int my_rank, int mpi_size, int startRowId, int endRowId);

/**
 * Print thread status at init (master/worker & rank)
 * @param my_rank, rank of thread
 * @param mpi_size, total nb of threads
 */
void printThreadStatus(int my_rank, int mpi_size);

/**
 * Init world1 & world2 for each thread
 * @param my_rank, rank of thread
 * @param world1, pointer to w1 array
 * @param world2, pointer to w2 array
 * @param gameMode, the type of world to init
 */
void initWorlds(int my_rank, unsigned int **world1, unsigned int **world2, int gameMode);




#endif //TP4_GAMEOFLIFE_MPI_H
