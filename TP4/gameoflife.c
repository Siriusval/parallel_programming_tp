
/*
 * Conway's Game of Life
 *
 * A. Mucherino
 *
 * PPAR, TP4
 *
 */

#ifdef _WIN32
#include <windows.h> //For windows Sleep()
#endif

#ifdef linux
#include <unistd.h> //For linux sleep()
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gameoflife.h"

int N = 32;
int itMax = 20;

unsigned int *allocate() {
    return (unsigned int *) calloc(N * N, sizeof(unsigned int));
}

int code(int x, int y, int dx, int dy) {
    int i = (x + dx) % N;
    int j = (y + dy) % N;
    if (i < 0) i = N + i;
    if (j < 0) j = N + j;
    return j * N + i;
}

void write_cell(int x, int y, unsigned int value, unsigned int *world) {
    int k;
    k = code(x, y, 0, 0);
    world[k] = value;
}

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


unsigned read_cell(int x, int y, int dx, int dy, unsigned int *world) {
    int k = code(x, y, dx, dy);
    return world[k];
}

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
}

short newGeneration(unsigned int *world1, unsigned int *world2, int ystart, int yend) {
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
    //Only the rows starting from xstart (incl.) to xend (excl.) (multithreading)
    for (y = ystart; y < yend; y++) {
        //for each column
        for (x = 0; x < N; x++) {

            //Get oldValue of cell
            unsigned int oldValue = read_cell(x, y, 0, 0, world1);
            unsigned int newValue = -1;

            //Check neighbors
            neighbors(x, y, world1, &nn, &n1, &n2);
            int liveNeighbors = n1+n2;
            //if cell live
            if(oldValue != EMPTY_CELL){
                //underpopulation, DIE! (< 2 neighbors)
                //overpopulation, DIE! (> 3 neighbors)
                if(liveNeighbors <2 || liveNeighbors > 3){
                    newValue = EMPTY_CELL;
                }
                //Live on to next gen (2 or 3 neighbors)
                else if (liveNeighbors == 2 || liveNeighbors == 3){
                    newValue = oldValue;
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
            else {
                newValue = EMPTY_CELL;
            }

            //Check if world changed
            if(oldValue != newValue){
                change = 1;
            }

            //Write
            if(newValue != EMPTY_CELL){ //World initialised at empty
                write_cell(x,y,newValue,world2);
            }

        }
    }

    return change;
}

void cls() {
    /*
    int i;
    for (i = 0; i < 10; i++) {
        fprintf(stdout, "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    }
     */
    system("cls");
}

void print(const unsigned int *world) {
    int i;
    cls();

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
    #ifdef _WIN32
    Sleep(500);
    #endif

    #ifdef linux
    sleep(500);
    #endif
}

/*
int main(int argc, char **argv) {

    //Init random generator
    srand(time(NULL));

    //INIT VARS
    int it, change;
    unsigned int *world1, *world2;
    unsigned int *worldaux;

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
 */
