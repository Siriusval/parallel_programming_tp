#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>
#include <math.h>

int N = 32; // The size of the array must be a power of 2
int n = 4; // The number of potential threads must divide N

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

/**
 * Sequential computation of sum of all the elements in the array
 * @param T, the array
 * @return , the sum of all the elements
 */
int sumTab(int* T){

    int sum = 0;

    // This loop can not be done in parallel
    // All iterations write at the same place
    for(int i = 0; i < N; i++){
        sum += T[i];
    }
    return(sum);
} // data aggregation (sum)

/**
 * First parallelizable version: pointer jumping<br>
 * sum of all the elements in the array
 * @param T, the array
 * @return , the sum of all the elements
 */
int sumTabPar(int *T)
{
    //For all steps
    for(int step = 0; step < log2(N);step++){
        printf("STEP %d\n", step);

        //For i in all proc
        for(int i = 0; i < N; i+=  (int) pow(2,step+1)){
            T[i] += T[i+ (int) pow(2,step)];
            printf("%d\n", T[i]);
        }
        printf("\n");
    }

    return(T[0]);
}

/**
 *  Second parallelizable version: partial aggregation<br>
 *  sum of all the elements in the array<br>
 * /!\ With only 4 threads
 * @param T, the array
 * @return , the sum of all the elements
 */
int sumTabPar2(int *T)
{

    //For each thread
    for(int i =0; i<n;i++){
        int rootIndex = (i * N/n);
        //Sum a specific part
        for(int j=1;j<N/n;j++){
            T[rootIndex] += T[rootIndex+j];
        }

    }

    //Sum separate parts
    int sum = 0;

    for(int i= 0; i<n;i++){
        sum += T[i * (N/n)];
    }

    return(sum);
}


int main(int argc,char *argv[])
{

    int *T = (int *)malloc(N * sizeof(int));
    initTab(T);
    printTab(T);

    printf("\n=== SUM - SEQ VERSION ===\n");
    printf("Sum of elements of T (seq): %d\n", sumTab(T));

    initTab(T);
    printf("\n=== SUM - PAR / PTR JUMPING VERSION ===\n");
    printf("Sum of elements of T (par - ptr jumping): %d\n", sumTabPar(T));

    initTab(T);
    printf("\n=== SUM - PAR / PARTIAL AGG VERSION ===\n");
    printf("Sum of elements of T (par - partial agg): %d\n", sumTabPar2(T));

}


