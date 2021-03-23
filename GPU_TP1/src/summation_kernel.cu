
// GPU kernel

/**
Compute a part of the  elements of the serie
@param data_out, array where to store results (1 cell per thread)
@param n, size of array
*/
__global__ void summation_kernel(float * data_out, int n)
{
    /*
    threadIdx.x -> id of thread in block
    blockIdx.x -> id of block
    blockDim.x -> nb of threads per block
    gridDim.x -> nb of blocks in grid
    */

  
    //Id of array, globally
    int threadIdGlobal = threadIdx.x + (blockIdx.x * blockDim.x);
 
    //check out of bound (as thread always multiple of 32, there can be unused threads)
    if(threadIdGlobal < n){
        //number of elements to compute, in case there's more elements than total threads
        int nbElemPerThread = n/(blockDim.x*gridDim.x);

        //bounds for i( ex: from i= 1024 to i=2048)
        int startIndex = threadIdGlobal * nbElemPerThread;
        int endIndex = startIndex + nbElemPerThread;

        //init
        data_out[threadIdGlobal] = 0.0;

        for(int  i = startIndex; i < endIndex ; i++){

            //Store all the computed values in the cell (1 cell per thread in data_out)
            data_out[threadIdGlobal] += pow(-1.0,i) / (i+1.0);
        }
    }
   
}

