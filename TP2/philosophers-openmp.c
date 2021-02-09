//
// Created by Valou on 09/02/2021.
//

//#include <unistd.h> //for linux sleep
#include <windows.h> //for windows Sleep()
#include <time.h> //for time()
#include <stdio.h> // printf()
#include <omp.h>

#define NB_PHILO 5

/** State of philosopher*/
enum Philo_State {THINKING = 0, HUNGRY = 1, EATING = 2};
/** Array of state representing each philosopher */
static enum Philo_State philo_states[NB_PHILO]={0};
/** Array of omp_lock_t representing each fork */
static omp_lock_t fork_states[NB_PHILO];

/**
 * Init the array of forks, create and init each lock
 */
void initLocks(){
    for(int i = 0; i< NB_PHILO; i++){
        omp_lock_t lock;
        omp_init_lock(&lock);
        fork_states[i] = lock;
    }
}

/**
 * Input a philosopher id, and get the id of the fork at his left
 * @param philoNb, the id of the philosopher
 * @return  the left fork id
 */
int getLeftForkIndex(int philoNb){
    if(philoNb == 0){
        return NB_PHILO-1;
    }
    return philoNb-1;
}

/**
 * Input a philosopher id, and get the id of the fork at his right
 * @param philoNb, the id of the philosopher
 * @return  the right fork id
 */
int getRightForkIndex(int philoNb){
    if(philoNb == NB_PHILO -1){
        return 0;
    }
    return philoNb+1;
}

/**
 * Get fork when available
 * @param forkId , the id of the fork to get
 */
void getFork(int forkId){
    //acquire resource
    omp_set_lock(&fork_states[forkId]);
}

/**
 * Put back fork when finished
 * @param forkId , the id of the fork to put
 */
void putFork( int forkId){
    //release resource
    omp_unset_lock(&fork_states[forkId]);
}

/**
 * Philosopher impl by Dijkstra
 * Each philosopher pick the left fork first, then the right one...
 * BUT ONE, the philosopher 0 always pick the right one first
 * @param i, the id of the philosopher to simulate
 */
void dijkstra(int i){
    //Init random var, for random sleep in eat and think
    int r;

    //Init id of forks on his sides
    int leftForkId ;
    int rightForkId;

    //Invert for philosopher number 0
    if(i == 0) { //INVERT
        leftForkId= getRightForkIndex(i);
        rightForkId = getLeftForkIndex(i);
    }
    else {
        leftForkId= getLeftForkIndex(i);
        rightForkId = getRightForkIndex(i);
    }

    //Loop
    while(1){
        switch (philo_states[i]) {
            case THINKING:
                r = rand()%5 + 1; //Eat randomly from 1 to 5s
                Sleep(r*1000);
                philo_states[i] = HUNGRY;
                printf("Philo %d HUNGRY\n",i);
                break;
            case HUNGRY:
                //grab forks
                getFork(leftForkId);
                printf("Philo %d got fork %d\n",i, leftForkId);
                getFork(rightForkId);
                printf("Philo %d got fork %d\n",i, rightForkId);
                philo_states[i] = EATING;
                printf("Philo %d EATING\n",i);
                break;
            case EATING:
                r = rand()%5 + 1; //Think randomly from 1 to 5s
                Sleep(r*1000);
                //Put fork down
                putFork(leftForkId);
                printf("Philo %d put fork %d\n",i, leftForkId);
                putFork(rightForkId);
                printf("Philo %d put fork %d\n",i, rightForkId);
                philo_states[i] = THINKING;
                printf("Philo %d THINKING\n",i);
                break;
        }
    }
}


int main() {

    srand(time(NULL));
    int i;
    initLocks();

    #pragma omp parallel shared(philo_states,fork_states)
    {
        #pragma omp for
        for(i = 0; i< NB_PHILO; i++){
            dijkstra(i);
        }
    }

    return 0;
}