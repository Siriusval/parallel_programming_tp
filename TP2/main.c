#include "omp.h"
#include <stdio.h>
int main()
{
    omp_set_dynamic(0);
    //omp_set_num_threads(4);
    #pragma omp parallel num_threads(4);
    {
        int id = omp_get_thread_num();
        printf("Hello (%d)", id );
        printf("World (%d)\n", id );
    }
    return 0;
}