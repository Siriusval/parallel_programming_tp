#define EMPTY 0
#define RED 1
#define BLUE 2
#define NB_NEIGHBORS 8


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
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    //get line
    y = (unsigned int)(y + dy) % domain_y;
    
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
    neighbors[2] = read_cell(source_domain,tx,ty,1,ty-1,domain_x,domain_y);
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
__device__ void writeValue(int* dest_domain, int tx, int ty, int domain_x, int myself){
    int id = getCellId(tx,ty,domain_x);
    dest_domain[id] = myself;
}

/**
 * Compute kernel
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
    writeValue(dest_domain,tx,ty,domain_x,myself);
}

