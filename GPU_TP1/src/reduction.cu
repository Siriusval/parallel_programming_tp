/**
Device function for reduction
sum all elements
@param arr, array of elements to sum
*/
__global__ void reduce(float* arr)
{
    //thread id
	const int tid = threadIdx.x;

    //Step size, start at 1, then double at each iteration
	int stepSize = 1;

    //Number of threads reducing, start at max, then halve at each iteration
	int numThreads = blockDim.x;

    //While there's still elems to reduce
	while (numThreads > 0)
	{
        //Only use necessary threads
		if (tid < numThreads)
		{
            /*
            First elem
            tid -> start at index tid (thread id)
            2 -> get one elem out of two
                at start we have n elements and n/2 threads
            stepSize -> at each iteration, go further to get element         
            */
			int first = tid * stepSize * 2;
			int second = first + stepSize;
            
            //sum 2 elems
			arr[first] += arr[second];
		}

        //double
		stepSize <<= 1;

        //halve
		numThreads >>= 1;
        __syncthreads();

	}
}

/**
Reduce an array of float
return sum of elements
@param arr, array to reduce
@param n, size of array
*/
float floatReduction(float* arr, int n)
{
    //Size for malloc
	size_t size = n * sizeof(float);
	
    //GPU Var
    float* d_ptr;
	
    //GPU Alloc
	cudaMalloc((void **)&d_ptr, size);
    //GPU Copy
	cudaMemcpy(d_ptr, arr, size, cudaMemcpyHostToDevice);

    //Device code
    reduce<<<1, n / 2 >>>(d_ptr); //TOFIX : nb max threads

    //Get results back
	float result;
	cudaMemcpy(&result, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);

    //Clean
	cudaFree(d_ptr);

	return result;
}