#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]){
	int n = 0,c = 0, i = 0;
	FILE * fp;
	if(argc>2){
		n=atoi(argv[1]);
		c=atoi(argv[2]);
		fp = fopen (argv[3],"w");
		srand(time(0)); 
		for(i = 0; i < n; i++){
       fprintf (fp, "%d ", rand()%(c/2)+1);
   }
   fprintf(fp, "\n");
   srand(time(0)); 
   for(i = 0; i < n; i++){
       fprintf (fp, "%d ", rand()%100+1);
   }
   fprintf(fp, "\n%d\n%d", n ,c);
   fclose (fp);
	}
        
}
