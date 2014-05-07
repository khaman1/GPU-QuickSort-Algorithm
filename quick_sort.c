#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
int DEBUG = 1;
char *data;

void readfile(char *filename, char *buffer, int num){

    FILE *fh;
    fh = fopen(filename, "r");
    fread(buffer, 1, num, fh);
    buffer[num] = '\0';
    fclose(fh);
}

int * get_list(int len){

    int *suffix_list = (int *) malloc(len*sizeof(int));
    int i;

    for(i=0; i<len; i++){
        suffix_list[i] = i;         
    }
    return suffix_list;
}

void quicksort(int* x, int first, int last){
    int pivot,j,i;
	float temp;

     if(first<last){
         pivot=first;
         i=first;
         j=last;

         while(i<j){
             while(x[i]<=x[pivot]&&i<last)
                 i++;
             while(x[j]>x[pivot])
                 j--;
             if(i<j){
                 temp=x[i];
                  x[i]=x[j];
                  x[j]=temp;
             }
         }

         temp=x[pivot];
         x[pivot]=x[j];
         x[j]=temp;
         quicksort(x,first,j-1);
         quicksort(x,j+1,last);

    }
}
  
void print_suffix_list(int *list, int len){
    int i=0;
    for(i=0; i<len; i++){
        printf("%d", list[i]);
        if(i != (len - 1)) printf(" ");
    }
    printf("\n");
}

int main(int argc, char *argv[]){
	clock_t start, end;
	double runTime;


    if(argc != 3){
        printf("Usage: ./quicksort -num -filename \n");
        exit(-1);
    }
    
    int num = atoi(argv[1]);
    char *filename = argv[2];

	start = clock();
    data = (char *) malloc((num+1)*sizeof(char));
    readfile(filename, data, num);

    int data_len = strlen(data);

    int *suffix_list = get_list(strlen(data));
    quicksort(suffix_list, 0, data_len-1);
    //print_suffix_list(suffix_list, data_len);
    
	end = clock();
	free(data);

	runTime = (end - start) / (double) CLOCKS_PER_SEC ;
	printf("%d %f\n", num, runTime);
}


