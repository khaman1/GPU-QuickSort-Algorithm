#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "io.h"

int main(int argc, char ** argv){
  int i;

  if(argc == 1){
    printf("Please specify at least one floating point array to operate upon!\n");
    exit(1);
  }

  int nlists = argc - 1;
  Data * data = (Data *) malloc(sizeof(Data));
  if(data == NULL){
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }
  char ** lists;
  lists = (char **) malloc(sizeof(char *) * nlists);
  if(lists == NULL){
    fprintf(stderr, "Unable to allocated memory\n");
    exit(1);
  }
  for(i = 0; i < nlists; i++){
    int len = strlen(argv[i+1]);
    lists[i] = (char *) malloc((sizeof(char)*len) + 1);
    if(lists[i] == NULL){
      fprintf(stderr, "Unable to allocate memory\n");
      exit(1);
    }
    strcpy(lists[i], argv[i+1]);
  }
  parser(data, nlists, lists);

  //  print_data(data, 0); //uncomment to check that data was read in properly

  Results * results = (Results *) malloc(sizeof(Results));
  if(results == NULL){
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  results->ntests = 2;
  results->nlists = data->nlists;
  results->rlist = (Result **) malloc(sizeof(Result *) * nlists);
  if(results->rlist == NULL){
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }
  for(i = 0; i < nlists; i++){
    results->rlist[i] = (Result *) malloc(sizeof(Result)*results->ntests);
    if(results->rlist[i] == NULL){
      fprintf(stderr, "Out of Memory\n");
      exit(1);
    }
  }

  for(i = 0; i < data->nlists; i++){
    //quicksort(data->fp_arr[i], data->length[i], &(results->rlist[0][i]));
    radixsort(data->fp_arr[i], data->length[i], &(results->rlist[1][i]));
    /* Execute the sorting algorithms here, each one on data->fp_arr[i] */
  }

  free(lists);
  free(data);
  free(results);

  return 0;
}
