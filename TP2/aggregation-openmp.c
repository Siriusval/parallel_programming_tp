
#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <math.h>
#include <omp.h>
#include <time.h>

int N = 64; // The size of the array must be a power of 2
int n = 4; // The number of potential threads must divide N

void printThreadInfo(){
    int nbThreads = omp_get_num_threads();
    printf("Nb threads : %d \n",nbThreads);

    int id = omp_get_thread_num();
    printf("Thread id : %d \n",id);
}


void printTab(int* T){

    for(int i = 0; i < N; i++)
    {
        printf("%d ", T[i]);
    }
    printf("\n");
} // printTab

void initTab(int* T){

    for(int i = 0; i < N; i++){
        T[i] = i;
    }
} // initTab

int sumTabSeq(int* T){

    int sum = 0;

    #pragma omp parallel
    {
        #pragma omp for
        for(int i = 0; i < N; i++){
            sum += T[i];
        }
    }

    return(sum);
} // data aggregation (sum)


int sumTabPar(int *T)
{
    int distToNext;

    for(int nbSteps = 0; nbSteps < log2(N); nbSteps++){
        distToNext = pow((float)2, nbSteps);

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < N; i += (int) pow(2, nbSteps + 1)) {
                T[i] += T[i + distToNext];
            }
        }
        printf("T after step %d: ", nbSteps);
        printTab(T);
    }
    printf("\n");
    return (T[0]);
}



int sumTabPar2(int *T)
{

    for(int i = 0; i < n; i++){
        int rootIndex = (i * N/n);

        #pragma omp parallel for
        {
            for (int j = 1; j < N/n; j++){
                T[rootIndex] += T[rootIndex + j];
            }
        }

    }

    printf("T before final aggregation: ");
    printTab(T);


    int sum = 0;
    for(int i = 0; i < n; i++)
    {
        sum += T[i * (N/n)];
    }

    return(sum);
}

int sumTabParOpenMp(int *T){

    // To be implemented
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int i =0; i<N;i++){
        sum += T[i];
    }
    return(sum);
}

int main(int argc,char *argv[])
{

    int *T = (int *)malloc(N * sizeof(int));
    initTab(T);
    printTab(T);

    printf("\n=== SUM - SEQ VERSION ===\n");
    printf("Sum of elements of T (seq): %d\n", sumTabSeq(T));

    initTab(T);
    printf("\n=== SUM - PTR JUMPING VERSION ===\n");
    printf("Sum of elements of T (par - ptr jumping): %d\n", sumTabPar(T));

    initTab(T);
    printf("\n=== SUM - PARTIAL AGG VERSION ===\n");
    printf("Sum of elements of T (par - partial agg): %d\n", sumTabPar2(T));

    initTab(T);
    printf("\n=== SUM - OPEN MP AGG VERSION ===\n");
    printf("Sum of elements of T (par - partial agg): %d\n", sumTabParOpenMp(T));


}


