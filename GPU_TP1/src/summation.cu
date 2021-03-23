#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "summation.h"
#include "summation_kernel.cu"
#include "reduction.cu"

/**
 * Compute Sum with i from n to 0
 */
float fromNTo0(int n){
    float sum = 0.0;

    //for each element 
    for(int i=n-1; i>=0;i--){
        sum += pow(-1.0,i)/(i+1.0);
    }

    return sum;
}
/**
 * Compute Sum with i from 0 to n
 */
float from0ToN(int n) {
    float sum = 0.0;

     //for each element 
    for(int i=0; i<n;i++){
        sum += pow(-1.0,i)/(i+1.0);
    }

    return sum;
}

// CPU implementation
float log2_series(int n)
{
    return from0ToN(n);
    //return fromNTo0(n);
}

/**
Run the computation & sum of the serie with the CPU
@param n, size of array
*/
float runCPUVersion(int n) {
    double start_time,end_time;
    float log2;

    start_time = getclock();
    log2 = log2_series(n);
    end_time = getclock();

    printf("CPU result: %f\n", log2);
    printf(" log(2)=%f\n", log(2.0));
    printf(" time=%fs\n", end_time - start_time);

    return log2;
}

/**
Reduce array, sum all elements in a sequential way
@param input, array to reduce
@param n, size of array
*/
float reduceSequential(float* input,int n){
    double start_time,end_time;
	float sum = 0.0;
    
    start_time = getclock();
    for(int i = 0; i < n; i++){
        sum += input[i];
    }
    end_time = getclock();

    printf("Sequential reduction result: %f\n", sum);
    printf(" time=%fs\n", end_time - start_time);

    return sum;
}

/**
Reduce array, sum all elements in a parallel way (CUDA)
@param input, array to reduce
@param n, size of array
*/
float reduceCuda(float* input,int n){
    double start_time,end_time;
    float sum;

    start_time = getclock();
    sum = floatReduction(input,n);
    end_time = getclock();

    printf("Cuda reduction result: %f\n", sum);
    printf(" time=%fs\n", end_time - start_time);

    return sum;
}

/**
Run the computation & sum of the serie with the GPU
@param n, size of array
*/
float runGPUVersion(int n){
    // Parameters definition <<<b,t>>>
    int blocks_in_grid = 8;//8;
    int threads_per_block = 4*32;//4 * 32;

    printf("INFOS:\n");
    //Find ideal gird & block size
    cudaOccupancyMaxPotentialBlockSize(&blocks_in_grid,&threads_per_block, summation_kernel,0,n);
    printf("\tblocks_in_grid: %d\n",blocks_in_grid);
    printf("\tthreads_per_block: %d\n",threads_per_block);

    // Round up according to array size 
    blocks_in_grid = (n + threads_per_block - 1) / threads_per_block; 
    printf("\tblocks_in_grid_rounded_up: %d\n",blocks_in_grid);

    // Alloc space for host (CPU) copy of output
    int num_threads = threads_per_block * blocks_in_grid;
    printf("\tnum_threads: %d\n\n",num_threads);
	size_t outputSize = num_threads * sizeof(float);
    float* output = (float *)malloc(outputSize);

	//Alloc space for device (GPU) copies of DATA_SIZE, output
    //int *d_data_size;
    float *d_output;
    //cudaMalloc((void **)&d_data_size, sizeof(int));
    cudaMalloc((void **)&d_output,outputSize);

    // Copy from CPU mem to GPU mem (dest, src, size, type)
    //cudaMemcpy( d_data_size, &data_size, sizeof(int), cudaMemcpyHostToDevice);

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Execute function <<<block,threads>>> (! async !)
    summation_kernel<<<blocks_in_grid,threads_per_block>>>(d_output,n);
    
    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // Get results back to CPU mem (blocking)
    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    //Finish REDUCTION
    float sum,sumCuda;
    
    //reduce sequential
    sum = reduceSequential(output,num_threads);

    //reduce cuda
    sumCuda = reduceCuda(output,num_threads);

    // Cleanup
    //cudaFree(d_data_size);
    cudaFree(d_output);
    free(output);

    //Print results
    printf("GPU results:\n");
    printf(" Sum: %f\n", sum);
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms

    double total_time = elapsedTime / 1000.;	// s
    double time_per_iter = total_time / (double) n;
    double bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Total time: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",
    	total_time,
    	time_per_iter * 1.e9,
    	bandwidth / 1.e9);
  
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

    return sum;
}


int floatCompare(float a, float b, float epsilon)
{
  return fabs(a - b) < epsilon;
}

void tests(){
    printf("====================\n");
    printf("        TESTS       \n");
    printf("====================\n");

   
    //from0ToN (log2_series), fromNTo0
    printf("====================\n");
    printf("     FROM 0 TO N    \n");
    printf("====================\n");
    
    float a, b;

    for(int i = 0; i<1000;i++){
        a = from0ToN(i);
        b = fromNTo0(i);
        assert(floatCompare(a,b,2E-6));
    }
    printf("from0ToN: %f\n",a);
    printf("fromNTo0: %f\n",b);
    printf("----------> from0ToN [OK]\n");
    printf("----------> fromNTo0 [OK]\n");
    printf("\n");
    
    //runCPUVersion
    printf("====================\n");
    printf("   RUN CPU VERSION  \n");
    printf("====================\n");

    a = runCPUVersion(1024*1024);
    assert(floatCompare(a,log(2.0),1E-3));

    printf("----------> runCPUVersion [OK]\n");
    printf("\n");

    //reduceSequential
    printf("====================\n");
    printf(" REDUCE SEQUENTIAL  \n");
    printf("====================\n");
    size_t size = 8;
    float fArray[]= {0.1,0.5,0.05,0.005,0.0005,0.00005,0.000005,0.0000005};
    a = reduceSequential(fArray,size);
    assert(floatCompare(a,0.6555555,1E-6));
    printf("----------> reduceSequential [OK]\n");
    printf("\n");
    

    //reduceCuda
    printf("====================\n");
    printf("     REDUCE CUDA    \n");
    printf("====================\n");
    a = reduceCuda(fArray,size);
    assert(floatCompare(a,0.6555555,1E-6)); //warning, do not try with non divisible
    printf("----------> reduceCuda [OK]\n");
    printf("\n");


    //runGPUVersion
    printf("====================\n");
    printf("   RUN GPU VERSION  \n");
    printf("====================\n");
    a = runGPUVersion(1024); //TODO : FIX nb block /threads
    assert(floatCompare(a,log(2.0),1E-3));
    printf("----------> runGPUVersion [OK]\n");

    printf("====================\n");
    printf("      END TESTS     \n");
    printf("====================\n");

}

int main(int argc, char ** argv){
    
    tests();
    // RUN CPU VERSION
    runCPUVersion(DATA_SIZE);
    
    // RUN GPU VERSION
    runGPUVersion(DATA_SIZE);
   
    return 0;
}

