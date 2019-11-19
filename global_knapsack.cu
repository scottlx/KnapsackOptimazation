#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "cuPrintf.cu"
#include "cuPrintf.cuh"

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}



#define PRINT_TIME         1
#define FILE_NAME "test.txt"
#define NUM_BAG 10
#define GRID_WIDTH 1

#define IMUL(a, b) __mul24(a, b)


void LoadInput(char file[], 
  int* num_painting, int* num_bags,
  int** weights, int** values){
FILE *infile; 
char filename[64];
int i, temp_int;
snprintf(filename, sizeof(filename), "%s", file);

// Open input txt file with error check 
infile = fopen (filename, "r"); 
if (infile == NULL) {
fprintf(stderr, "\nError opening file\n"); 
exit (1); 
} 

// Read integers

// load number of painting in the first line
fscanf (infile, "%d", &temp_int);
*num_painting = temp_int;

// load number of bags in the first line
fscanf (infile, "%d", num_bags);


// prepare memory for weights and values array
*weights = (int *) malloc((*num_painting)*sizeof(int));
*values = (int *) malloc((*num_painting)*sizeof(int));

// load weights
for(i=0; i<(*num_painting); i++){
  fscanf (infile, "%d", &temp_int);
  (*weights)[i] = temp_int;
  }

  // load values
  for(i=0; i<(*num_painting); i++){
  fscanf (infile, "%d", &temp_int);
  (*values)[i] = temp_int;
  }
  fclose (infile); 

  printf("number of bags = %d\n", *num_bags);
  printf("number of paintings = %d\n", *num_painting);

}

void Worker(int n, int b, int* weight, int* value, int* result){
  int i, j;
  int* tmp1 = (int *) malloc(b*sizeof(int));
  int* tmp2 = (int *) malloc(b*sizeof(int));
  int* tmp3;
  for(j=0; j<b; j++){
    if (weight[0] > j) {
      tmp1[j] = 0;
    } 
    else {
      tmp1[j] = value[0];
    }
  }

  for(i = 1; i<n; i++){
    //printf("i = %d\n",i);
    for(j=0; j<b; j++){
      //printf("j = %d\n",j);
      if (j < weight[i] || tmp1[j] >= tmp1[j-weight[i]] + value[i]){
        //printf("er\n");
        tmp2[j] = tmp1[j];
      }
      else{
        tmp2[j] = tmp1[j-weight[i]] + value[i];
      }
    }
    tmp3 = tmp1;
    tmp1 = tmp2;
    tmp2 = tmp3;
  }
  for(j=0; j<b; j++){
    result[j] = tmp1[j];
  }
}




void print_result(int num_bags, int* result){
  int j;
    for(j=0; j < num_bags;j++){
      printf("%d ",result[j]);
    }
    printf("\n");
}




__global__ void kernel (int p, int b, int* w, int* v, int* r) {
	
	// int col = threadIdx.x;
	int row = threadIdx.y;
  int i;
  __shared__ int last_col[NUM_BAG];
  __shared__ int this_col[NUM_BAG];
  //initialize first element
  if (w[0] > row) {
    last_col[row] = 0;
  } 
  else {
    last_col[row] = v[0];
  }
  __syncthreads();
	for(i=1; i<p; i++){
    if (row < w[i] || last_col[row] >= last_col[row-w[i]] + v[i]){
      this_col[row] = last_col[row];
    }
    else{
      this_col[row] = last_col[row-w[i]] + v[i];
    }
    __syncthreads();
    last_col[row] = this_col[row];
  }
  r[row]=this_col[row];
}



int main(int argc, char **argv){
  //num of paintings and bags
  int num_painting,num_bags;

  //some temp variables
  int i, temp_int;
  
  //store cpu time
  struct timeval end, begin;

  // Arrays on the host memory
  int* weights;
  int* values;
  int* results;
  int* results_gold;
  // load input from txt
  
  FILE *infile; 
  char filename[64] = FILE_NAME;

  // Open input txt file with error check 
  infile = fopen (filename, "r"); 
  if (infile == NULL) {
    fprintf(stderr, "\nError opening file\n"); 
    exit (1); 
  } 

  // Read integers
  // load number of painting in the first line
  fscanf (infile, "%d", &temp_int);
  num_painting = temp_int;

  // load number of bags in the first line
  fscanf (infile, "%d", &temp_int);
  num_bags = temp_int;


  // prepare memory for weights and values array
  weights = (int *) malloc(num_painting*sizeof(int));
  values = (int *) malloc(num_painting*sizeof(int));

  // load weights
  for(i=0; i<num_painting; i++){
    fscanf (infile, "%d", &temp_int);
    weights[i] = temp_int;
    }

    // load values
    for(i=0; i<num_painting; i++){
    fscanf (infile, "%d", &temp_int);
    values[i] = temp_int;
    }
    fclose (infile); 

    printf("number of bags = %d\n", num_bags);
    printf("number of paintings = %d\n", num_painting);


   // Allocate arrays on host memory
   results = (int *) malloc(num_bags * sizeof(int));
   results_gold = (int *) malloc(num_bags * sizeof(int));
  
  // GPU Timing variables
  cudaEvent_t start, stop;
  float elapsed_gpu;

  // Arrays on GPU global memoryc
  int *gpu_weights;
  int *gpu_values;
  int *gpu_results;

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate GPU memory
  size_t allocSize_1 = num_painting * sizeof(int);
  size_t allocSize_2 = num_bags * sizeof(int);
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_weights, allocSize_1));
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_values, allocSize_1));
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_results, allocSize_2));

  printf("Allocate done\n\n");
  
  
  // Compute the results on the host

  gettimeofday(&begin, NULL);
  Worker(num_painting, num_bags, weights, values, results_gold);
  gettimeofday(&end, NULL);

  printf("cpu time =  %lu us\n", (end.tv_sec - begin.tv_sec) * 1000000 + end.tv_usec - begin.tv_usec);
  printf("cpu_result:\n");
	print_result(num_bags, results_gold);

#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  
#endif

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(gpu_weights, weights, allocSize_1, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(gpu_values, values, allocSize_1, cudaMemcpyHostToDevice));

	// init cuPrint
	cudaPrintfInit ();
	
	dim3 dimGrid(GRID_WIDTH,GRID_WIDTH,1);
	dim3 dimBlock(1,num_bags,1);

  cudaEventRecord(start, 0);
  // Launch the kernel
  kernel<<<dimGrid, dimBlock>>>(num_painting, num_bags, gpu_weights, gpu_values, gpu_results);
  cudaEventRecord(stop,0);
  // end of cuPrint
  cudaPrintfDisplay (stdout, true);
	cudaPrintfEnd ();

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(results, gpu_results, allocSize_2, cudaMemcpyDeviceToHost));

#if PRINT_TIME
  // Stop and destroy the timer
  
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  printf("\nGPU time: %f (usec)\n", elapsed_gpu * 1000);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif

  printf("gpu_result:\n");
  print_result(num_bags, results);
	

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(gpu_weights));
  CUDA_SAFE_CALL(cudaFree(gpu_values));
  CUDA_SAFE_CALL(cudaFree(gpu_results));

  free(weights);
  free(values);
  free(results);
  free(results_gold);

  return 0;
}
