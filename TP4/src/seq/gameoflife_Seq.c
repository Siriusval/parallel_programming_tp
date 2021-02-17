#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../lib/inc/gameoflife.h"

int main() {

    printf("GAME OF LIFE / SEQUENTIAL\n");
    //Init random generator
    srand(time(NULL));

    //INIT VARS
    int it, change;
    unsigned int *world1, *world2;
    unsigned int *worldAux;

    int input = askGameMode();

    //CREATE WORLD
    world1 = createWorld(input);

    world2 = allocate();
    print(world1);

    //Loop for itMax generations
    it = 0;
    change = 1;
    while (change && it < itMax) {
        change = newGeneration(world1, world2, 0, N);
        worldAux = world1;
        world1 = world2;
        world2 = worldAux;
        print(world1);
        it++;
    }

    // ending
    free(world1);
    free(world2);
    return 0;

}
