
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>

const int NB_CARS = 10;
const int NB_PLACES = 4;
const int MAX_SPEED = 20;

struct Car {
   int id;
   int speed;
   int place;
};  

void initCar(struct Car *c, int n){

    c->id = n;
    c->speed = rand() % MAX_SPEED;
    c->place = rand() % NB_PLACES;
    
}

void printCars(struct Car cars [NB_CARS]){
    
    for(int i = 0; i < NB_CARS; i++){
        printf("Car #%d --- speed: %d, place: %d\n", cars[i].id, cars[i].speed, cars[i].place);
    }
    
}

int main()
{

    // init
    struct Car cars [NB_CARS];
    srand(time(NULL));
    for(int i = 0; i < NB_CARS; i++){
        initCar(&cars[i], i);
        
    }
    
    printCars(cars);
    
    return 0;
}


