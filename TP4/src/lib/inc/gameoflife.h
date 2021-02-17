/**
 * @author Valou
 */
#ifndef TP4_GAMEOFLIFE_H
#define TP4_GAMEOFLIFE_H

#define EMPTY_CELL 0
#define TYPE_O_CELL 1
#define TYPE_X_CELL 2

/* X IS THE HORIZONTAL AXIS, AKA. COLUMN NUMBER !
 * Y IS THE VERTICAL AXIS, AKA. ROW NUMBER
 */

/** Size of world (matrix NxN) */
extern int N;
/** Number of iterations */
extern int itMax;
/** Sleep duration in ms */
extern int sleepDurationMs;

/**
 * Create a world depending on user input
 * @param input, the game mode
 * @return the world corresponding to the input
 */
unsigned int * createWorld(int input);

/**
 * Ask number to user
 * @return a number between 0 and 3 (incl.)
 */
int getNumber();

/**
 * Ask game mode to the user
 * @return the int corresponding to the game mode
 */
int askGameMode();

/**
 * Fill the world with Empty cells
 * @param world, the world array to clean
 */
void cleanWorld(unsigned int *world);


/**
 * Allocate memory for N*N matrix
 * @return the pointer to the allocated array
 */
unsigned int *allocate();

/**
 * Get cell id (from the current one)
 * @param x the column of the current cell
 * @param y the row of the current cell
 * @param dx the horizontal offset, -1, 0 or 1
 * @param dy the vertical offset, -1, 0 or 1
 */
int code(int x, int y, int dx, int dy);

/**
 * Set a new value in the cell
 * @param x, the column of the cell
 * @param y, the row of the cell
 * @param value, the new value of the cell
 * @param world, the array that contains all cells
 */
void write_cell(int x, int y, unsigned int value, unsigned int *world);

/**
 * Generate a world with random values in cells
 * @return the pointer to the world array
 */
unsigned int *initialize_random();

/**
 * Generate a world for testing
 * Only one cell
 * @return pointer to the array containing the world
 */
unsigned int *initialize_dummy();

/**
 * Generate a world with a glider
 * @return pointer to the array containing the world
 */
unsigned int *initialize_glider();

/**
 * Generate a world with a "small explorer"
 * @return pointer to the array containing the world
 */
unsigned int *initialize_small_exploder();
/**
 * Read the value of a cell (from the current cell)
 * @param x, the column of the current cell
 * @param y, the row of the current cell
 * @param dx, the horizontal offset of the cell to read
 * @param dy, the vertical offset of the cell to read
 * @param world the world containing the cells
 * @return the value of the cell to read
 */
unsigned read_cell(int x, int y, int dx, int dy, unsigned int *world);

/**
 * Count the number of cells around the current cell
 * Update the counters passed in parameter
 * @param x, column of the current cell
 * @param y, row of the current cell
 * @param dx, horizontal offset of cell to look for
 * @param dy, vertical offset of cell to look for
 * @param world, the array containing the cells
 * @param nn, the counter for number of empty cells around
 * @param n1, the counter for number of type "o" cells around
 * @param n2, the counter for number of type "x" cells around
 */
void update(int x, int y, int dx, int dy, unsigned int *world, int *nn, int *n1, int *n2);

/**
 * Count the neighbors around the current cell
 * @param x, column of the current cell
 * @param y, row of the current cell
 * @param world, the array containing the cells
 * @param nn, the counter for number of empty cells around
 * @param n1, the counter for number of type "o" cells around
 * @param n2, the counter for number of type "x" cells around
 */
void neighbors(int x, int y, unsigned int *world, int *nn, int *n1, int *n2);

/**
 * Computing a new generation
 * Create a new world with the new values for the cells
 * @param world1, old world
 * @param world2, current world (start empty)
 * @param yStart, id of starting row to compute (incl.)
 * @param yEnd, id of ending row to compute (excl.)
 * @return change,  if the world has changed
 */
short newGeneration(unsigned int *world1, unsigned int *world2, int yStart, int yEnd);

/**
 * Clear console/screen
 */
void cls();

/**
 * Display the world (in console)
 * @param world, the array containing the cells
 */
void print(const unsigned int *world);

#endif //TP4_GAMEOFLIFE_H
