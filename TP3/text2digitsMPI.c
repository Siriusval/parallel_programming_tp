
/* Sequential algorithm for converting a text in a digit sequence
 *
 * PPAR, TP3
 *
 * A. Mucherino
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

int main(int argc,char *argv[])
{
    int i,text_size;
    int count,ascii_code;
    char *text;
    char filename[20];
    short notblank,notpoint,notnewline;
    short number,minletter,majletter;
    FILE *input;
    int  my_rank,mpi_size,start,end,step; //Added for MPI

    //MPI Init
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);

    printf("EXERCISE A\n");

    strcpy(filename,argv[1]);
    // getting started (we suppose that the argv[1] contains the filename related to the text)
    input = fopen(filename,"r");
    if (!input)
    {
        fprintf(stderr,"%s: impossible to open file '%s', stopping\n",argv[0],filename);
        return 1;
    }
    else{
         printf("File %s opened.\n",filename);
    }

    // checking file size
    fseek(input,0,SEEK_END);
    text_size = ftell(input);
    rewind(input);

    // reading the text
    text = (char*)calloc(text_size + 1, sizeof(char));
    for (i = 0; i < text_size; i++) text[i] = fgetc(input);

    //split text for each mpi part
    step = text_size / mpi_size;
    start = step*my_rank;

    //master code

    //init data
    //Distribute

    //Collect


    //WORKER CODE
    //Start index
    if (my_rank != 0){ //for all but first
        while(text[start] != 32 && start < text_size){ // move start index to next blank space (don't start in middle of word)
            start++;
        }
    }
    //end index
    if (my_rank != mpi_size-1){ //for all worker but the last one
        end = step*(my_rank+1);
        while (text[end] != 32 && end < text_size){// move end index to next blank space (don't start in middle of word)
            end++;
        }
    }
    else{ //for last worker only
        end = text_size;
    }

    //receive
    //compute
    //Send back results


    // converting the text
    count = 0;
    printf("%d",my_rank);
    for (i = start; i < end; i++) {
         ascii_code = (int) text[i];
         notblank = (ascii_code != 32);
         notpoint = (ascii_code != 46);
         notnewline = (ascii_code != 10);
         number = (ascii_code >= 48 && ascii_code <= 57);  // 0-9
         majletter = (ascii_code >= 65 && ascii_code <= 90);  // A-Z
         minletter = (ascii_code >= 97 && ascii_code <= 122);  // a-z

         //if valid char, count +1
         if(majletter || minletter){
              count++;
         }
         else {
              if(number){
                    printf("%c",text[i]);
              }
              //if special char, print 0
              if(notblank && notpoint && notnewline){
                    printf("%d",0);
              }

              //if counting a word before special char
              if(count != 0){
                    printf("%d",count);
                    count =0;
              }
         }
    }
    if(count != 0){
      printf("%d",count);
    }
    printf("\n");


    // closing
    MPI_Finalize();
    free(text);
    fclose(input);

    // ending
    return 0;
}

