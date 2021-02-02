/*
 * PPAR, TP1
 *
 * for Exercises 1.E and 2.A
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int maxn = 100000;

int main(int argc,char *argv[])
{
   int i,j,n;
   int T[maxn];
   double val[maxn],res[maxn];
   struct timeval time1,time2;
   int time;

   // getting the tree size
   if (argc < 2)
   {
      fprintf(stderr,"%s: please provide the size of the tree : ",argv[0]);
      fscanf(stdin,"%d",&n);
   }
   else
   {
      n = atoi(argv[1]);
   };
   if (n <= 0)
   {
      fprintf(stderr,"%s: invalid value for the tree size, size is %d!\n",argv[0],n);
      return 1;
   };

   // randomly generating the tree
   T[0] = 0;  val[0] = 0.0;
   for (i = 1; i < n; i++)
   {
      T[i] = rand()%i;
      val[i] = 0.1*(int)(10.0*(double)rand()/RAND_MAX);
   };
   printf("%s: the tree: ",argv[0]);
   if (n < 50)
   {
      printf("\n"); 
      for (i = 0; i < n; i++)  printf("T(%d-->%d) %5.2lf\n",i,T[i],val[i]);
   }
   else
   {
      printf("suppressed information (n = %d)\n",n);
   };

   // checking the time
   gettimeofday(&time1,0);
   
   // finding the branch weights
   for (i = 0; i < n; i++)
   {
      // to be filled
   };
   printf("%s: found solutions: ",argv[0]);
   if (n < 50)
   {
      printf("\n");
      for (i = 0; i < n; i++)  printf("For branch (%d---->%d), the weight is %5.2lf\n",0,i,res[i]);
   }
   else
   {
      printf("suppressed information (n = %d)\n",n);
   };

   // checking the time
   gettimeofday(&time2,0);
   time = (time2.tv_sec * 1000000 + time2.tv_usec) - (time1.tv_sec * 1000000 + time1.tv_usec);
   printf("%s: time = %d\n",argv[0],time);

   return 0;
};


