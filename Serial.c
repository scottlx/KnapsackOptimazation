#include <stdio.h>
#include <stdlib.h>

void LoadInput(char file[], 
							 int* num_painting, int* num_bags,
							 int* weights, int* values){
		FILE *infile; 
		char filename[64];
		int i;
		snprintf(filename, sizeof(filename), "%s", file);
   
    // Open input txt file with error check 
    infile = fopen (filename, "r"); 
    if (infile == NULL) { 
        fprintf(stderr, "\nError opening file\n"); 
        exit (1); 
    } 
    
    // Read integers
    
  	fscanf (infile, "%d", &i);    
  	while (!feof (infile)){  
      printf ("%d ", i);
      fscanf (infile, "%d", &i);      
    }

    fclose (infile); 
}

void Worker(int* weight, int* values, int* result, int n, int b){
		int i, j;
		for(j=0; j<b; j++){
			if (weight[0] > j) {
	  		result[0][j] = 0;
				} 
			else {
	  			result[0][j] = values[0];
			}
		}
		for(i = 1; i<n; i++){
			for(j=0; j<b; j++){
				if (j < weight[i] || result[i-1][j] >= result[i-1][j-weight[i]] + value[i])){
					result[i][j]=result[i-1][j];
				}
				else{
					result[i][j] = result[i-1][j-weight[i]] + value[i];
				}
			}
		}
}


int main(int argc, char *argv[]){
		int num_painting,num_bags;
    int *weights;
    int *values;
    int *result;
		LoadInput("test1.txt",&num_painting, &num_bags, weights, values);
		result = malloc(sizof(int[num_painting][num_bags]));
		Worker(weights, values, result, num_painting, num_bags);
		return 0; 
}
