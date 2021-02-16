
/*
 * Conway's Game of Life
 *
 * A. Mucherino
 *
 * PPAR, TP4
 *
 */

#include <stdio.h>
#include <stdlib.h>
//#include <unistd.h> //For linux sleep()
#include <windows.h> //For windows Sleep()
#include <time.h>
#include "mpi.h"

#define EMPTY_CELL 0
#define TYPE_O_CELL 1
#define TYPE_X_CELL 2

/** Size of world (matrix NxN) */
int N = 32;
/** Number of iterations */
int itMax = 20;

/**
 * Allocate memory for N*N matrix
 * @return the pointer to the allocated array
 */
unsigned int *allocate() {
    return (unsigned int *) calloc(N * N, sizeof(unsigned int));
}

/**
 * Get cell id (from the current one)
 * @param x the coordinate x of the current cell
 * @param y the coordinate y of the current cell
 * @param dx the horizontal offset, -1, 0 or 1
 * @param dy the vertical offset, -1, 0 or 1
 */
int code(int x, int y, int dx, int dy) {
    int i = (x + dx) % N;
    int j = (y + dy) % N;
    if (i < 0) i = N + i;
    if (j < 0) j = N + j;
    return i * N + j;
}

// writing into a cell location
/**
 * Set a new value in the cell
 * @param x, the x coordinate of the cell
 * @param y, the y coordinate of the cell
 * @param value, the new value of the cell
 * @param world, the array that contains all cells
 */
void write_cell(int x, int y, unsigned int value, unsigned int *world) {
    int k;
    k = code(x, y, 0, 0);
    world[k] = value;
}

/**
 * Generate a world with random values in cells
 * @return the pointer to the world array
 */
unsigned int *initialize_random() {
    int x, y;
    unsigned int cell;
    unsigned int *world;

    //Create word
    world = allocate();

    //For each column
    for (x = 0; x < N; x++) {
        //For each line
        for (y = 0; y < N; y++) {
            //Empty cell
            if (rand() % 5 != 0) {
                cell = EMPTY_CELL;
            }
            //o cell
            else if (rand() % 2 == 0) {
                cell = TYPE_O_CELL;
            }
            //x cell
            else {
                cell = TYPE_X_CELL;
            }
            write_cell(x, y, cell, world);
        }
    }
    return world;
}

/**
 * Generate a world for testing
 * Only one cell
 * @return pointer to the array containing the world
 */
unsigned int *initialize_dummy() {
    int x, y;
    unsigned int *world;

    world = allocate();
    for (x = 0; x < N; x++) {
        for (y = 0; y < N; y++) {
            write_cell(x, y, x % 3, world);
        }
    }
    return world;
}

/**
 * Generate a world with a glider
 * @return pointer to the array containing the world
 */
unsigned int *initialize_glider() {
    int x, y, mx, my;
    unsigned int *world;

    world = allocate();
    for (x = 0; x < N; x++) {
        for (y = 0; y < N; y++) {
            write_cell(x, y, EMPTY_CELL, world);
        }
    }

    mx = N / 2 - 1;
    my = N / 2 - 1;
    x = mx;
    y = my + 1;
    write_cell(x, y, TYPE_O_CELL, world);
    x = mx + 1;
    y = my + 2;
    write_cell(x, y, TYPE_O_CELL, world);
    x = mx + 2;
    y = my;
    write_cell(x, y, TYPE_O_CELL, world);
    y = my + 1;
    write_cell(x, y, TYPE_O_CELL, world);
    y = my + 2;
    write_cell(x, y, TYPE_O_CELL, world);

    return world;
}

/**
 * Generate a world with a "small explorer"
 * @return pointer to the array containing the world
 */
unsigned int *initialize_small_exploder() {
    int x, y, mx, my;
    unsigned int *world;

    world = allocate();
    for (x = 0; x < N; x++) {
        for (y = 0; y < N; y++) {
            write_cell(x, y, EMPTY_CELL, world);
        }
    }

    mx = N / 2 - 2;
    my = N / 2 - 2;
    x = mx;
    y = my + 1;
    write_cell(x, y, TYPE_X_CELL, world);
    x = mx + 1;
    y = my;
    write_cell(x, y, TYPE_X_CELL, world);
    y = my + 1;
    write_cell(x, y, TYPE_X_CELL, world);
    y = my + 2;
    write_cell(x, y, TYPE_X_CELL, world);
    x = mx + 2;
    y = my;
    write_cell(x, y, TYPE_X_CELL, world);
    y = my + 2;
    write_cell(x, y, TYPE_X_CELL, world);
    x = mx + 3;
    y = my + 1;
    write_cell(x, y, TYPE_X_CELL, world);

    return world;
}


/**
 * Read the value of a cell (from the current cell)
 * @param x, the x coordinate of the current cell
 * @param y, the y coordinate of the current cell
 * @param dx, the horizontal offset of the cell to read
 * @param dy, the vertical offset of the cell to read
 * @param world the world containing the cells
 * @return the value of the cell to read
 */
unsigned read_cell(int x, int y, int dx, int dy, unsigned int *world) {
    int k = code(x, y, dx, dy);
    return world[k];
}

/**
 * Count the number of cells around the current cell
 * Update the counters passed in parameter
 * @param x, horizontal coordinate of the current cell
 * @param y, vertical coordinate of the current cell
 * @param dx, horizontal offset of cell to look for
 * @param dy, vertical offset of cell to look for
 * @param world, the array containing the cells
 * @param nn, the counter for number of empty cells around
 * @param n1, the counter for number of type "o" cells around
 * @param n2, the counter for number of type "x" cells around
 */
void update(int x, int y, int dx, int dy, unsigned int *world, int *nn, int *n1, int *n2) {
    unsigned int cell = read_cell(x, y, dx, dy, world);
    if (cell != EMPTY_CELL) {
        (*nn)++;
        if (cell == TYPE_O_CELL) {
            (*n1)++;
        } else {
            (*n2)++;
        }
    }
}


/**
 * Count the neighbors around the current cell
 * @param x, horizontal coordinate of the current cell
 * @param y, vertical coordinate of the current cell
 * @param world, the array containing the cells
 * @param nn, the counter for number of empty cells around
 * @param n1, the counter for number of type "o" cells around
 * @param n2, the counter for number of type "x" cells around
 */
void neighbors(int x, int y, unsigned int *world, int *nn, int *n1, int *n2) {
    int dx, dy;

    (*nn) = 0;
    (*n1) = 0;
    (*n2) = 0;

    // same line
    dx = -1;
    dy = 0;
    update(x, y, dx, dy, world, nn, n1, n2);
    dx = +1;
    dy = 0;
    update(x, y, dx, dy, world, nn, n1, n2);

    // one line down
    dx = -1;
    dy = +1;
    update(x, y, dx, dy, world, nn, n1, n2);
    dx = 0;
    dy = +1;
    update(x, y, dx, dy, world, nn, n1, n2);
    dx = +1;
    dy = +1;
    update(x, y, dx, dy, world, nn, n1, n2);

    // one line up
    dx = -1;
    dy = -1;
    update(x, y, dx, dy, world, nn, n1, n2);
    dx = 0;
    dy = -1;
    update(x, y, dx, dy, world, nn, n1, n2);
    dx = +1;
    dy = -1;
    update(x, y, dx, dy, world, nn, n1, n2);
};

/**
 * Computing a new generation
 * Create a new world with the new values for the cells
 * @param world1, old world
 * @param world2, current world (start empty)
 * @param xstart, id of starting column to compute
 * @param xend, id of ending column to compute
 * @return change,  if the world has changed
 */
short newGeneration(unsigned int *world1, unsigned int *world2, int xstart, int xend) {
    int x, y;
    int nn, n1, n2;
    unsigned int cell;
    short change = 0;

    // cleaning destination world
    for (x = 0; x < N; x++) {
        for (y = 0; y < N; y++) {
            write_cell(x, y, EMPTY_CELL, world2);
        }
    }

    // generating the new world
    //Only the columns starting from xstart to xend (multithreading)
    for (x = xstart; x < xend; x++) {
        //for each line
        for (y = 0; y < N; y++) {

            //Get oldValue of cell
            unsigned oldValue = read_cell(x, y, 0, 0, world1);
            unsigned newValue = oldValue;

            //Check neighbors
            neighbors(x, y, world1, &nn, &n1, &n2);
            int liveNeighbors = n1+n2;

            //if cell live
            if(oldValue != EMPTY_CELL){
                //underpopulation, DIE! (< 2 neighbors)
                if(liveNeighbors <2){
                    newValue = EMPTY_CELL;
                }
                //Live on to next gen (2 or 3 neighbors)
                else if (liveNeighbors == 2 || liveNeighbors == 3){
                    newValue = oldValue;
                }
                //overpopulation, DIE! (> 3 neighbors)
                else if (liveNeighbors > 3){
                    newValue = EMPTY_CELL;
                }
            }
            //If cell empty & 3 neighbors, come alive
            else if(liveNeighbors == 3){
                if(n1>=n2){
                    newValue = TYPE_O_CELL;
                }
                else {
                    newValue = TYPE_X_CELL;
                }
            }
            //Else empty cell
            else{
                newValue = EMPTY_CELL;
            }

            //Write
            write_cell(x,y,newValue,world2);

            //Check if world changed
            if(oldValue != newValue){
                change = 1;
            }
        }
    }
    return change;
}

/**
 * Clear console/screen
 */
void cls() {
    int i;
    for (i = 0; i < 10; i++) {
        fprintf(stdout, "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    }
}

/**
 * Display the world (in console)
 * @param world, the array containing the cells
 */
void print(const unsigned int *world) {
    int i;
    //cls();
    system("cls");

    for (i = 0; i < N; i++) fprintf(stdout, "-");

    for (i = 0; i < N * N; i++) {
        if (i % N == 0) fprintf(stdout, "\n");
        if (world[i] == EMPTY_CELL) fprintf(stdout, " ");
        if (world[i] == TYPE_O_CELL) fprintf(stdout, "o");
        if (world[i] == TYPE_X_CELL) fprintf(stdout, "x");
    }
    fprintf(stdout, "\n");

    for (i = 0; i < N; i++) fprintf(stdout, "-");
    fprintf(stdout, "\n");
    Sleep(500);
}

/**
 * Main
 * @param argc, argument count (size of argument array)
 * @param argv, argument vector (array containing them)
 */
int main(int argc, char *argv[]) {

    //INIT VARS
    int it, change;
    unsigned int *world1, *world2;
    unsigned int *worldaux;
    int my_rank, mpi_size, start, end, step; //Added for MPI

    //MPI Init
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);

    //printf("My rank : %d\n",my_rank);
    //if(my_rank ==0) printf("MPI size : %d\n",mpi_size);

    //CREATE WORLD
    // world1 = initialize_dummy();
    //world1 = initialize_random();
    world1 = initialize_glider();
    //world1 = initialize_small_exploder();
    world2 = allocate();
    print(world1);

    //Loop for itMax generations
    it = 0;
    change = 1;
    while (change && it < itMax) {
        change = newGeneration(world1, world2, 0, N);
        worldaux = world1;
        world1 = world2;
        world2 = worldaux;
        print(world1);
        it++;
    }

    // ending
    free(world1);
    free(world2);
    return 0;
}

