#include <stdio.h>
#include <stdlib.h>

#define W_RANGE 1000


// # of painting = first arg, file name = second arg
int main(int argc, char *argv[]){
        int n = 0, i = 0;
        long int w_sum = 0;
        long int w = 0;
        long int c =0;
        FILE * fp;
        if(argc>1){
                n=atoi(argv[1]);
                fp = fopen (argv[2],"w");
                fprintf(fp, "%ld\n", n);
                srand(time(0));

                for(i = 0; i < n; i++){
                   w = rand()%W_RANGE+1;
                   w_sum += w;
       fprintf (fp, "%d ", w);
   }
   c = w_sum /2;
   fprintf(fp, "\n%ld\n", c);

                for(i = 0; i < n; i++){
       fprintf (fp, "%d ", rand()%(c/2)+1);
        }
   fprintf(fp, "\n");
   srand(time(0));



   fclose (fp);
        }

}
