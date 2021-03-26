#define EMPTY 0
#define RED 1
#define BLUE 2
#define NB_NEIGHBORS 8
#define DOMAIN_X 128
#define DOMAIN_Y  128
#define STEPS 2
#define THREADS_PER_BLOCK 64
#define BLOCKDIM_X 64
#define BLOCKDIM_Y 1
#include <assert.h>


/**
 * Get the cell id in 1D array
 * @param x, col index
 * @param y, line index
 * @param domain_x, the width of a line
 *
 * @return the id of the cell(x,y) in the 1D array
*/
__device__ int getCellId(int x, int y, int domain_x){
    return (y * domain_x + x);
}
/**
 * Reads neighbor cell, at (x+dx, y+dy)
 * @param source_domain, array containing GoL state
 * @param x, col of current cell
 * @param y, line of current cell
 * @param dx, col offset 
 * @param dy, line offset 
 * @param domain_x, domain width
 * @param domain_y, domain height
 * 
 * @return value of neighbor cell 
 */
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy, unsigned int domain_x, unsigned int domain_y)
{
    //get col
    x = (unsigned int)((x + dx + domain_x) % domain_x);    // Wrap around
    //get line
    y = (unsigned int)((y + dy + domain_y) % domain_y);
    
    //go to line y (y* domain_x), and to the xth cell
    int id = getCellId(x,y,domain_x);
    return source_domain[id];
}

/**
* Compute the values of the 8 neighbors of cell (x,y)
* Put them in neighbors array
* @param neighbors, the array to fill with values
* @param source_domain, the array representing GoL
* @param tx, the thread col index
* @param ty, the thread line index
* @param domain_x, domain width
* @param domain_y, domain height
*/
__device__ void getNeighborsValues(int*neighbors, int* source_domain,int tx, int ty,int domain_x,int domain_y){
    //Read all the neighbors to construct the array
    //top line
    neighbors[0] = read_cell(source_domain,tx,ty,-1,-1,domain_x,domain_y);
    neighbors[1] = read_cell(source_domain,tx,ty,0,-1,domain_x,domain_y);
    neighbors[2] = read_cell(source_domain,tx,ty,1,-1,domain_x,domain_y);
    //same line
    neighbors[3] = read_cell(source_domain,tx,ty,-1,0,domain_x,domain_y);
    neighbors[4] = read_cell(source_domain,tx,ty,1,0,domain_x,domain_y);
    //bottom line
    neighbors[5] = read_cell(source_domain,tx,ty,-1,1,domain_x,domain_y);
    neighbors[6] = read_cell(source_domain,tx,ty,0,1,domain_x,domain_y);
    neighbors[7] = read_cell(source_domain,tx,ty,1,1,domain_x,domain_y);
}

/**
 * Count the number of red and blue cells in an array
 * @param neighbors, the array containing all neighbors values
 * @param nbRed, pointer to store int value (number of red cells in neigbors)
 * @param nbBlue, pointer to store int value (number of blue cells in neigbors)
 */
__device__ void countRedBlueValues(int* neighbors, int* nbRed, int* nbBlue){
    for(int i = 0; i < NB_NEIGHBORS; i++){
        if(neighbors[i]==RED){
            *nbRed += 1;
        }
        else if(neighbors[i]==BLUE){
           *nbBlue += 1;
        }
    }
}

/**
 * Compute the new value of current cell,
 * depending on how much red and blue cells are around
 * @param myself, current cell value
 * @param nbRed, number of red cells in neighborhood
 * @param nbBlue, number of blue cells in neighborhood
 * 
 */
__device__ void newValueOfCurrentCell(int* myself,int nbRed, int nbBlue){
    //if myself is a dead cell
    if(*myself==EMPTY){
        //Birth conditions
        if(nbBlue + nbRed == 3){
            if(nbBlue > nbRed){
                //myself becomes blue
                *myself=BLUE;
            }
            else{
               //myself becomes red
                *myself=RED;
            }
        }
    }
    //else it's red or blue
    else {

        //Death conditions
        if(nbRed + nbBlue < 2 || nbRed + nbBlue > 3){
            //Die
            *myself = EMPTY;
        }
    }
}

/**
 * Write newValue of cell at its correct position in domain
 * @param dest_domain, destination domain to write cell value
 * @param tx, thread col index
 * @param ty, thread line index
 * @param domain_x, width of domain
 * @param myself, cell value to write
 */
__device__ void writeValue(int* dest_domain,int gindex, int myself){
    dest_domain[gindex] = myself;
}

/**
 * Copy all the needed neighbors into temp
 * Ex : if blockDim = (2,2), copy (4,4) portion in shared mem
 * @param source_domain, domain where we take our values
 * @param domain_x, width of domain
 * @param domain_y, height of domain
 * @param tx, thread global col index (2D)
 * @param ty, thread global row index (2D)
 * @param gindex, thread global index (1D)
 * @param temp, domain where we put our values (shared mem)
 * @param tempDimX, width of shared mem
 * @param tempDimY, height of shared mem
 * @param localIndexX, thread local col index (2D)
 * @param localIndexY, thread local row index (2D)
 * @param lindex, thread local index (1D)
 */
__device__ void copyNeighborsIntoTemp(int* source_domain,int domain_x, int domain_y, int tx, int ty, int gindex,int* temp, int tempDimX,int tempDimY,int localIndexX, int localIndexY, int lindex){
    
   
    //Copy neighbors
    int isTopRow = localIndexY == 1;
    int isBottomRow = localIndexY == tempDimY -2;
    int isLeftCol = localIndexX == 1;
    int isRightCol = localIndexX ==  tempDimX -2;
   

    temp[lindex] = source_domain[gindex];

    int index;
    //if on top
    if(isTopRow)
    {
        //copy top neighbor
        index = getCellId(localIndexX,localIndexY-1,tempDimX); 
        temp[index]= read_cell(source_domain,tx,ty,0,-1,domain_x,domain_y);
    }

    //if on bottom
    if(isBottomRow)
    {
        //copy bottom neighbor
        index = getCellId(localIndexX,localIndexY+1,tempDimX);
        temp[index]=  read_cell(source_domain,tx,ty,0,1,domain_x,domain_y);
    }

    //if on left
    if(isLeftCol)
    {
        //copy left neighbor
        index = getCellId(localIndexX-1,localIndexY,tempDimX);
        temp[index]=  read_cell(source_domain,tx,ty,-1,0,domain_x,domain_y);

    }
    //if on right
    if(isRightCol)
    {
        //copy right neighbor
        index = getCellId(localIndexX+1,localIndexY,tempDimX);
        temp[index]=  read_cell(source_domain,tx,ty,1,0,domain_x,domain_y);

    }

    //CORNER
    //if corner top left
    if(isTopRow && isLeftCol)
    {
        //copy top left neighbor
        index = getCellId(localIndexX-1,localIndexY-1,tempDimX);
        temp[index]=  read_cell(source_domain,tx,ty,-1,-1,domain_x,domain_y);
    }

    //if corner top right
    if(isTopRow && isRightCol)
    {
        //copy top right neighbor
        index = getCellId(localIndexX+1,localIndexY-1,tempDimX);
        temp[index]=  read_cell(source_domain,tx,ty,1,-1,domain_x,domain_y);

    }
    //if corner bottom left
    if(isBottomRow && isLeftCol)
    {
        //copy bottom left neighbor
        index = getCellId(localIndexX-1,localIndexY+1,tempDimX);
        temp[index]=  read_cell(source_domain,tx,ty,-1,1,domain_x,domain_y);

    }
    //if corner bottom right
    if(isBottomRow && isRightCol)
    {
        //copy bottom right neighbor
        index = getCellId(localIndexX+1,localIndexY+1,tempDimX);
        temp[index]=  read_cell(source_domain,tx,ty,1,1,domain_x,domain_y);

    }

    __syncthreads();


}

/**
 * Unit testing of all functions
 * Only need to test it with 1 thread
 * Call from main
 */
__global__ void test_kernel(){

    if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y==0){

        printf("TESTS\n");
        printf("\tSTARTED\n");
        //getCellId
        printf("\tgetCellId\n");
        int  array[] = {1,2,3,4,5,6,7,8,9};
        int dimX = 3;
        assert(getCellId(0,0,dimX) == 0);
        assert(getCellId(1,0,dimX) == 1);
        assert(getCellId(2,0,dimX) == 2);
        assert(getCellId(0,1,dimX) == 3);
        assert(getCellId(1,1,dimX) == 4);
        assert(getCellId(2,1,dimX) == 5);
        assert(getCellId(0,2,dimX) == 6);
        assert(getCellId(1,2,dimX) == 7);
        assert(getCellId(2,2,dimX) == 8);
        printf("\t---->getCellId[OK]\n");

        //read_cell
        printf("\tread_cell\n");
        assert(read_cell(array,0,0,0,0,dimX,dimX)==1);
        assert(read_cell(array,1,0,0,0,dimX,dimX)==2);
        assert(read_cell(array,2,0,0,0,dimX,dimX)==3);
        assert(read_cell(array,0,1,0,0,dimX,dimX)==4);
        assert(read_cell(array,1,1,0,0,dimX,dimX)==5);
        assert(read_cell(array,2,1,0,0,dimX,dimX)==6);
        assert(read_cell(array,0,2,0,0,dimX,dimX)==7);
        assert(read_cell(array,1,2,0,0,dimX,dimX)==8);
        assert(read_cell(array,2,2,0,0,dimX,dimX)==9);

        //go out of bound left
        assert(read_cell(array,0,0,-1,0,dimX,dimX)==3);
        //go out of bound top
        assert(read_cell(array,0,0,0,-1,dimX,dimX)==7);
        //go out of bound right
        assert(read_cell(array,2,2,1,0,dimX,dimX)==7);
        //go out of bound bot
        assert(read_cell(array,2,2,0,1,dimX,dimX)==3);
        printf("\t---->read_cell[OK]\n");

        //getNeighborsValues
        printf("\tgetNeighborsValues\n");
        int neighbors[NB_NEIGHBORS];
        getNeighborsValues(neighbors, array,1,1,dimX,dimX);
        assert(neighbors[0] == 1);
        assert(neighbors[1] == 2);
        assert(neighbors[2] == 3);
        assert(neighbors[3] == 4);
        assert(neighbors[4] == 6);
        assert(neighbors[5] == 7);
        assert(neighbors[6] == 8);
        assert(neighbors[7] == 9);
        printf("\t---->getNeighborsValues[OK]\n");

        //countRedBlueValues  
        printf("\tcountRedBlueValues\n");
        //test 1 
        int colors[] = {RED,RED,RED,RED,RED,RED,RED,RED};

        int red = 0;
        int blue = 0;
        countRedBlueValues(colors,&red,&blue);
        assert(red==8);
        assert(blue==0);

        //test 2
        red = 0;
        blue = 0;
        int colors2[]= {BLUE,BLUE,BLUE,BLUE,BLUE,BLUE,BLUE,BLUE};
        countRedBlueValues(colors2,&red,&blue);
        assert(red==0);
        assert(blue==8);
        printf("\t---->countRedBlueValues[OK]\n");
        
        //newValueOfCurrentCell 
        printf("\tnewValueOfCurrentCell\n");
        //EMPTY + 3 voisins, birth, majority red
        int myVal = EMPTY;
        newValueOfCurrentCell(&myVal,2,1);
        assert(myVal == RED);

        //EMPTY + 3 neightbors, birth, majority blue
        myVal = EMPTY;
        newValueOfCurrentCell(&myVal,1,2);
        assert(myVal == BLUE);

        //LIVE AND 2 OR 3 neighbors survive, keep same value
        myVal = 4;
        newValueOfCurrentCell(&myVal,0,2);
        assert(myVal == 4);

        myVal = 4;
        newValueOfCurrentCell(&myVal,3,0);
        assert(myVal == 4);

        myVal = 4;
        newValueOfCurrentCell(&myVal,1,1);
        assert(myVal == 4);

        //live and more than 3 neighbors die
        myVal = BLUE;
        newValueOfCurrentCell(&myVal,0,4);
        assert(myVal == EMPTY);
        myVal = BLUE;
        newValueOfCurrentCell(&myVal,4,0);
        assert(myVal == EMPTY);
        printf("\t---->newValueOfCurrentCell[OK]\n");

        //writeValue
        printf("\twriteValue\n");

        int array2[] = {0,0,0,0,0,0,0,0,0};
        for(int i = 0; i< 9; i++){
            writeValue(array2,i,i*2);
            assert(array2[i]==i*2);
        }
        printf("\t---->newValueOfCurrentCell[OK]\n");


        //copyNeighborsIntoTemp
        printf("\tcopyNeighborsIntoTemp\n");

        /*
        original domain is (4,4)
        thread dim is (2,2)
        temp domain is (4,4)
        check that temp is copied correctly
        */
        int srcDomain[] =     {1,2,3,4,
                                5,6,7,8,
                                9,10,11,12,
                                13,14,15,16};
        int srcDimX = 4;

        int temp[] =   {0,0,0,0,
                        0,0,0,0,
                        0,0,0,0,
                        0,0,0,0};

        int tempDimX= 4;

        int nbThread  = 2;
        //for each thread
        //each line
        for(int j = 0; j <nbThread;j++){
            //each col
            for(int i = 0; i <nbThread; i ++){
                int gindex = getCellId(i,j,srcDimX);
                int localIndexX = i+1;
                int localIndexY = j+1;
                int lindex = getCellId(localIndexX,localIndexY,tempDimX);
                copyNeighborsIntoTemp(srcDomain,srcDimX,srcDimX, i,j,gindex, temp,tempDimX, tempDimX,localIndexX,localIndexY,lindex);
            }
        }

        int res[] =   {16,13,14,15,
                        4,1,2,3,
                        8,5,6,7,
                        12,9,10,11};
                        
        for(int j = 0; j <srcDimX;j++){
            for(int i = 0; i <srcDimX; i++){
                int index = j*srcDimX+i;
               assert(temp[index]==res[index]);
               //printf("%d ",temp[index]);
            } 
            //printf("\n");
        }
        printf("\t---->copyNeighborsIntoTemp[OK]\n");

        //Check if test is called
        //assert(0);
        printf("TESTS FINISHED\n");


    }
}
/**
 * Compute kernel (with shared memory)
 * Compute the new state of GoL into dest_domain
 * @param source_domain, array for GoL current state
 * @param dest_domain, array to compute GoL next state 
 * @param domain_x, domain width
 * @param domain_y, domain height
 */
__global__ void life_kernel(int * source_domain, int * dest_domain,
    int domain_x, int domain_y)
{
    // get thread id
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    // if out of bound, thread can chill
    if(tx >= domain_x || ty >= domain_y) return;
    

    //Create shared memory
    //see call with <<< _, _, size >>>
    extern __shared__ int temp[];
    int tempDimX = BLOCKDIM_X+2;
    int tempDimY = BLOCKDIM_Y+2;

    int gindex = getCellId(tx,ty,domain_x); //global index
    int localIndexX = threadIdx.x+1;
    int localIndexY = threadIdx.y+1;
    int lindex = getCellId(localIndexX,localIndexY,tempDimX); 

    // Read input elements into shared memory
    //Copy elements in the middle of shared memory

    copyNeighborsIntoTemp(source_domain,domain_x,domain_y, tx,ty,gindex, temp,tempDimX, tempDimY,localIndexX,localIndexY,lindex);
    
    // Read current cell
    int myself = read_cell(temp, localIndexX, localIndexY, 0, 0,
                           tempDimX, tempDimY);
    
    // Read the 8 neighbors
    int neighbors[NB_NEIGHBORS];
    getNeighborsValues(neighbors,temp, localIndexX,localIndexY,tempDimX, tempDimY);

    //Count the numbers of red and blue neighbors
    int nbRed = 0;
    int nbBlue = 0;
    countRedBlueValues(neighbors,&nbRed,&nbBlue);

    // Compute new value of current cell
    newValueOfCurrentCell(&myself,nbRed, nbBlue);

    // Write it in dest_domain
    writeValue(dest_domain,gindex,myself);
}

/**
 * Compute kernel (with global memory)
 * Compute the new state of GoL into dest_domain
 * @param source_domain, array for GoL current state
 * @param dest_domain, array to compute GoL next state 
 * @param domain_x, domain width
 * @param domain_y, domain height
 */
__global__ void life_kernel_global(int * source_domain, int * dest_domain,
    int domain_x, int domain_y)
{
    // get thread id
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    // if out of bound, thread can chill
    if(tx >= domain_x || ty >= domain_y) return;
   
    int gindex = getCellId(tx,ty,domain_x); //global index
   
    // Read current cell
    int myself = read_cell(source_domain, tx, ty, 0, 0,
                           domain_x, domain_y);
    
    // Read the 8 neighbors
    int neighbors[NB_NEIGHBORS];
    getNeighborsValues(neighbors,source_domain, tx,ty,domain_x, domain_y);

    //Count the numbers of red and blue neighbors
    int nbRed = 0;
    int nbBlue = 0;
    countRedBlueValues(neighbors,&nbRed,&nbBlue);

    // Compute new value of current cell
    newValueOfCurrentCell(&myself,nbRed, nbBlue);

    // Write it in dest_domain
    writeValue(dest_domain,gindex,myself);
}
