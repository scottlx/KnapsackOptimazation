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
#define FILE_NAME "test2.txt"
#define MAX_THREAD 1024

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

void Worker(int n, int b, int* weight, int* value, int* result, int* decisionM){
  int i, j;
  int* tmp1 = (int *) malloc(b*sizeof(int));
  int* tmp2 = (int *) malloc(b*sizeof(int));
  int* tmp3;
  int weight_diff = 0;
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
      weight_diff = tmp1[j-weight[i]] + value[i];
      if (j < weight[i] || tmp1[j] >= weight_diff){
        //printf("er\n");
        tmp2[j] = tmp1[j];
      }
      else{
        tmp2[j] = weight_diff;
        decisionM[i/32*b+j] += 1 << (i%32);
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

__constant__ int constant_weights[4096];
__constant__ int constant_values[4096];

__global__ void kernel_initial_constant (int col,  int* global_decisionM, int* this_col) {
	
  int row = threadIdx.y+MAX_THREAD*blockIdx.y;
  if (constant_weights[0] > row) {
    this_col[row] = 0;
  } 
  else {
    this_col[row] = constant_values[0];
    global_decisionM[row] = 1;
  }
}

__global__ void kernel_initial_global (int col, int* weights, int* values, int* global_decisionM, int* this_col) {
	
  int row = threadIdx.y+MAX_THREAD*blockIdx.y;
  if (weights[0] > row) {
    this_col[row] = 0;
  } 
  else {
    this_col[row] = values[0];
    global_decisionM[row] = 1;
  }
}


__global__ void kernel_constant (int col, int* this_col,  int* global_decisionM, int* last_col) {
	
  int row = threadIdx.y+MAX_THREAD*blockIdx.y;
  if (row < constant_weights[col] || last_col[row] >= last_col[row-constant_weights[col]] + constant_values[col]){
    this_col[row] = last_col[row];
  }
  else{
    this_col[row] = last_col[row-constant_weights[col]] + constant_values[col];
    global_decisionM[row] += 1 << (col%32);
  }
}

__global__ void kernel_global (int col, int* weights, int* values, int* global_decisionM, int* this_col, int* last_col) {
	
  int row = threadIdx.y+MAX_THREAD*blockIdx.y;
  if (row < weights[col] || last_col[row] >= last_col[row-weights[col]] + values[col]){
    this_col[row] = last_col[row];
  }
  else{
    this_col[row] = last_col[row-weights[col]] + values[col];
    global_decisionM[row] += 1 << (col%32);
  }
}


int main(int argc, char **argv){
  //num of paintings and bags
  long int num_painting,num_bags, temp_long_int;

  //some temp variables
  int i, temp_int;
  
  //store cpu time
  struct timeval end, begin;

  // Arrays on the host memory
  int* weights;
  int* values;
  int* results;
  int* results_gold;
  int* decisionM_gold;
  int* decisionM;
  int* Zeros;
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
  fscanf (infile, "%ld", &temp_long_int);
  num_painting = temp_long_int;

 

  // prepare memory for weights and values array
  weights = (int *) malloc(num_painting*sizeof(int));
  values = (int *) malloc(num_painting*sizeof(int));
  

  // load weights
  	for(i=0; i<num_painting; i++){
    	fscanf (infile, "%d", &temp_int);
    	weights[i] = temp_int;
    }
    
    // load number of bags in the first line
  	fscanf (infile, "%ld", &temp_long_int);
	  num_bags = temp_long_int;

    // load values
    for(i=0; i<num_painting; i++){
    	fscanf (infile, "%d", &temp_int);
    	values[i] = temp_int;
    }

  decisionM = (int *) malloc(num_painting*num_bags/32*sizeof(int));
  decisionM_gold = (int *) malloc(num_painting*num_bags/32*sizeof(int));
  Zeros = (int *) malloc(num_bags*sizeof(int));
    for(i=0; i<num_bags; i++){
    	Zeros[i]=0;
    }

  for(i=0; i<num_bags*num_painting/32; i++){
    decisionM_gold[i] = 0;
  }

    
    fclose (infile); 


   // Allocate arrays on host memory
   results = (int *) malloc(num_bags * sizeof(int));
   results_gold = (int *) malloc(num_bags * sizeof(int));
  
  // GPU Timing variables
  cudaEvent_t start, stop;
  float elapsed_gpu;

  // Arrays on GPU global memoryc
  int *col_1;
  int *col_2;
  int *global_weights;
  int *global_values;
  int* global_decisionM;

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate GPU memory
  size_t allocSize_1 = num_painting * sizeof(int);
  size_t allocSize_2 = num_bags * sizeof(int);
  CUDA_SAFE_CALL(cudaMalloc((void **)&col_1, allocSize_2));
  CUDA_SAFE_CALL(cudaMalloc((void **)&col_2, allocSize_2));
  CUDA_SAFE_CALL(cudaMalloc((void **)&global_weights, allocSize_1));
  CUDA_SAFE_CALL(cudaMalloc((void **)&global_values, allocSize_1));
  CUDA_SAFE_CALL(cudaMalloc((void **)&global_decisionM, allocSize_2));
  

#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  
#endif

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(global_weights, weights, allocSize_1, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(global_values, values, allocSize_1, cudaMemcpyHostToDevice));
  
  if(num_painting<=4096){
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(constant_weights, weights, allocSize_1,0));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(constant_values, values, allocSize_1, 0));
  }

	// init cuPrint
	cudaPrintfInit ();
	
	dim3 dimGrid(1,(num_bags+MAX_THREAD-1)/MAX_THREAD,1);
  dim3 dimBlock(1,MAX_THREAD,1);
  CUDA_SAFE_CALL(cudaMemcpy(global_decisionM, Zeros, allocSize_2, cudaMemcpyHostToDevice));
  if(num_painting<=4096){
    cudaEventRecord(start, 0);
    // Launch the kernel
    //intial
    kernel_initial_constant<<<dimGrid, dimBlock>>>(i, global_decisionM, col_1);
    //start from second col
    for(i=1;i<num_painting;i++){
      if(i%2)
        kernel_constant<<<dimGrid, dimBlock>>>(i, global_decisionM, col_2, col_1);
      else
        kernel_constant<<<dimGrid, dimBlock>>>(i, global_decisionM, col_1, col_2);
      if(i%32==0){
        CUDA_SAFE_CALL(cudaMemcpy(decisionM + i/32*num_bags, global_decisionM, allocSize_2, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(global_decisionM, Zeros, allocSize_2, cudaMemcpyHostToDevice));
      }
    }
    cudaEventRecord(stop,0);
    // end of cuPrint
  }
  else{
    cudaEventRecord(start, 0);
    // Launch the kernel
    kernel_initial_global<<<dimGrid, dimBlock>>>(i, global_weights, global_values, global_decisionM, col_1);
    for(i=1;i<num_painting;i++){
      if(i%2)
        kernel_global<<<dimGrid, dimBlock>>>(i, global_weights, global_values, global_decisionM,col_2, col_1);
      else
        kernel_global<<<dimGrid, dimBlock>>>(i, global_weights, global_values, global_decisionM,col_1, col_2);
      if(i%32==0){
        CUDA_SAFE_CALL(cudaMemcpy(decisionM + i/32*num_bags, global_decisionM, allocSize_2, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(global_decisionM, Zeros, allocSize_2, cudaMemcpyHostToDevice));
      }
    }
    cudaEventRecord(stop,0);
    // end of cuPrint
  }
  #if PRINT_TIME
    // Stop and destroy the timer
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  #endif
  cudaPrintfDisplay (stdout, true);
	cudaPrintfEnd ();

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  if(num_painting%2){
    CUDA_SAFE_CALL(cudaMemcpy(results, col_1, allocSize_2, cudaMemcpyDeviceToHost));
  }
  else{
    CUDA_SAFE_CALL(cudaMemcpy(results, col_2, allocSize_2, cudaMemcpyDeviceToHost));
  }
  // Compute the results on the host

  gettimeofday(&begin, NULL);
  Worker(num_painting, num_bags, weights, values, results_gold, decisionM_gold);
  gettimeofday(&end, NULL);

  printf("testfile = %s, bag = %d, paintings = %d\n", FILE_NAME, num_bags, num_painting);
  printf("cpu time =  %lu us\n", (end.tv_sec - begin.tv_sec) * 1000000 + end.tv_usec - begin.tv_usec);
  printf("\nGPU time: %f (usec)\n\n", elapsed_gpu * 1000);
  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(global_weights));
  CUDA_SAFE_CALL(cudaFree(global_values));
  CUDA_SAFE_CALL(cudaFree(col_1));
  CUDA_SAFE_CALL(cudaFree(col_2));

  free(weights);
  free(values);
  free(results);
  free(results_gold);

  return 0;
}
