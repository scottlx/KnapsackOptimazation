#include <stdio.h>
#include <stdlib.h>

void LoadInput(char file[], 
							 int* num_painting, int* num_bags,
							 int* weights, int* values){
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
  	weights = malloc((*num_painting)*sizeof(int));
  	values = malloc((*num_painting)*sizeof(int));
  	
  	// load weights
  	for(i=0; i<(*num_painting); i++){
  		fscanf (infile, "%d", &temp_int);
  		weights[i] = temp_int;
  	}
  	
  	// load values
  	for(i=0; i<(*num_painting); i++){
  		fscanf (infile, "%d", &temp_int);
  		values[i] = temp_int;
  	}
    fclose (infile); 
    
    //printf("number of bags = %d\n", *num_bags);
    //printf("number of paintings = %d\n", *num_painting);
}





int main(int argc, char *argv[]){
		int num_painting, num_bags;
    int weights;
    int values;
		LoadInput("test.txt", &num_painting, &num_bags, &weights, &values);
		
		
		return 0; 
}
