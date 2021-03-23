#ifndef SUMMATION_H_   /* Include guard */
#define SUMMATION_H_

//number of elements to compute
#define DATA_SIZE (1024 * 1024 * 256 )

float fromNTo0(int n);

float from0ToN(int n);

float log2_series(int n);

float runCPUVersion(int n);

float reduceSequential(float* input,int n);

float reduceCuda(float* input,int n);

float runGPUVersion(int n);

int main(int argc, char ** argv);

#endif // FOO_H_
