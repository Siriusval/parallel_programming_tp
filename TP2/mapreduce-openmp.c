#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define NB_WORDS 10
#define STRING_MAX_SIZE 20
#define ALPHABET_SIZE 26

/* A function to generate random strings */
void gen_random(char *s, int len) {
    static const char alphanum[] = "abcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    s[len] = 0;
}

/* A function to print arrays of strings */
void printArrayStrings(char **array){
    
    for(int i = 0; i < NB_WORDS; i++){
        printf("%s\n", array[i]);
    }
}

/* A function to print the counts */
void printArrayInt(int *count){
    
    for(int i = 0; i < ALPHABET_SIZE; i++){
        printf("%d ", count[i]);
    }
    printf("\n");
}

/* A function that maps a String to a count of characters */
void map1 (char* c, int *count){

    //Init count array
    for(int i= 0; i < ALPHABET_SIZE; i++){
        count[i] = 0;
    }

    //parse word
    for(int i=0; i < strlen(c);i++){
        count[(int) c[i]-97] += 1;
    }

}

/*
 * Based on the output of the Map phase, compute the total number of occurrences
 * of each letter
 * Store the result in count[0]
 */
void reduce1(int **count) {
    # pragma omp parallel
    {
        #pragma omp for
        {
            for(int j =0; j < ALPHABET_SIZE ;j++){
            #pragma omp parallel for shared(count[0][j]) reduction (+ : count[0][j])
                {
                    for (int i = 1 ;i < NB_WORDS;i++){
                        count[0][j] += count[i][j];
                    }
                }
            }
        }
    };

}
/*
 *  Based on the output of the Reduce1 phase, compute the total number of characters in the text.
 *  Return sum, the number of letter in text
 */
int reduce2(int **count) {
    int sum = 0;

    #pragma omp parallel for shared(sum) reduction(+:sum)
    {
        for (int i = 0; i< ALPHABET_SIZE;i++){
            sum += count[0][i];
        }
    }

    return sum;

}


int main()
{

    // init
    char *array[NB_WORDS];
    srand(time(NULL));
    for(int i = 0; i < NB_WORDS; i++){
        int strSize = 1 + rand() % STRING_MAX_SIZE;
        array[i] = (char *)malloc(strSize * sizeof(char));
        gen_random(array[i], strSize);
    }
    
    printArrayStrings(array);

    //Map
    int *count[NB_WORDS];

    #pragma omp parallel for
    for (int i = 0 ;i < NB_WORDS;i++){
        // map (on a single word)
        count[i] = (int *)malloc(ALPHABET_SIZE * sizeof(int));
        map1(array[i], count[i]);
        printArrayInt(count[i]);
    }

    //Reduce1
    reduce1(count);
    printf("\nReduce 1 : \n");
    printArrayInt(count[0]);

    //Reduce 2
    int sum = reduce2(count);
    printf("reduce 2 : %d\n",sum);

    return 0;
}
