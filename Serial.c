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



int main(int argc, char *argv[]){
		int num_painting,num_bags;
    int weights;
    int values;
		LoadInput("test1.txt",&num_painting, &num_bags, &weights, &values);

		return 0; 
}
