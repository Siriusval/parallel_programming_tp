
#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define NB_CARS 20
#define NB_PLACES 4
#define MAX_SPEED 5

/**
 * Struct for car
 * Contain id, speed and place
 */
struct Car {
   int id;
   int speed;
   int place;
};  

/**
 * Init car item with values
 * @param c, the car to init
 * @param n, the id of the car
 */
void initCar(struct Car *c, int n){

    c->id = n;
    c->speed = rand() % MAX_SPEED;
    c->place = rand() % NB_PLACES;
    
}

/**
 * Print the data of list of cars
 * @param cars, the list of cars
 */
void printCars(struct Car cars [], int size){

    if (size == 0){
        printf("Nothing to display.\n");
        return;
    }

    for(int i = 0; i < size; i++){
        printf("Car #%d --- speed: %d, place: %d\n", cars[i].id, cars[i].speed, cars[i].place);
    }
    
}

/**
 * Get the array of cars with a speed of 0
 * @param cars
 */
int getStoppedCars(struct Car cars [NB_CARS], struct Car res [NB_CARS]){
    int count = 0;
    #pragma omp parallel
    {
        #pragma omp for
        {
            for(int i = 0; i < NB_CARS; i++){
                if(cars[i].speed == 0){

                    #pragma omp critical
                    {
                        res[count] = cars[i];
                        count++;
                    }
                }
            }
        }
    };

    return count;
}

/**
 * Get the array of cars at place 2
 * @param cars
 */
int getCarsAtPlace2(struct Car cars [NB_CARS], struct Car res [NB_CARS]){
    int count = 0;
    #pragma omp parallel
    {
        #pragma omp for
        {
            for(int i = 0; i < NB_CARS; i++){
                if(cars[i].place == 2){

                    #pragma omp critical
                    {
                        res[count] = cars[i];
                        count++;
                    }
                }
            }
        }
    };

    return count;
}

/**
 * Get the array of cars at place 2 & speed 0
 * @param cars
 */
int mergeStoppedAndPlace2(struct Car stopped [], int size1,struct Car place2 [], int size2,struct Car res [NB_CARS]){
    int count = 0;
    #pragma omp parallel
    {
        #pragma omp for
        for(int i =0; i< size1; i++){ //one thread for each car
            for(int j =0; j< size2; j++){ //search for the same id in the other array
                if(stopped[i].id == place2[j].id){
                    #pragma omp critical
                    {
                        res[count] = stopped[i];
                        count++;
                    }
                }
            }
        }
    };

    return count;
}



/**
 * Main
 * @return 0
 */
int main()
{

    // init
    struct Car cars [NB_CARS];
    srand(time(NULL));
    for(int i = 0; i < NB_CARS; i++){
        initCar(&cars[i], i);
        
    }
    
    printCars(cars, NB_CARS);
    printf("\n");

    //Stopped cars
    printf("Stopped cars\n");
    struct Car results1 [NB_CARS];
    int size1 = getStoppedCars(cars,results1);
    printCars(results1,size1);
    printf("\n");

    //Cars at place 2
    printf("Cars at place 2\n");
    struct Car results2 [NB_CARS];
    int size2 = getCarsAtPlace2(cars,results2);
    printCars(results2,size2);
    printf("\n");

    //Merge
    printf("Merged array\n");
    struct Car results3 [NB_CARS];
    int size3 = mergeStoppedAndPlace2(results1,size1,results2,size2,results3);
    printCars(results3,size3);
    
    return 0;
}


