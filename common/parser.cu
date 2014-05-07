#include <stdio.h>
#include <stdlib.h>
#include "io.h"

int load_array (Data * data, char * fname, int idx){
  FILE * list = fopen(fname, "r");
  if(list == NULL){
    fprintf(stderr, "Unable to open %s\n", fname);
    exit(1);
  }
  int meta = fgetc(list);
  if(meta != '#'){
    fprintf(stderr, "%s is not properly formatted\n", fname);
    exit(1);
  }
  int llen;
  fscanf(list, "%d", &llen);
  if(llen <= 0){
    fprintf(stderr, "%s is not properly formatted\n", fname);
    exit(1);
  }

  meta = fgetc(list);
  if(meta != '#'){
    fprintf(stderr, "%s is not properly formatted\n", fname);
    exit(1);
  }
  data->fp_arr[idx] = (float *) malloc(sizeof(float *) * llen);

  if(data->fp_arr[idx] == NULL){
    fprintf(stderr, "Unable to allocate space for %s in memory\n", fname);
    exit(1);
  }

  for(int i = 0; i < llen; i++){
    fscanf(list, "%f ", &(data->fp_arr[idx][i]));
  }

  return llen;
}

void parser(Data * data, int nlists, char ** names){
  data->nlists = nlists;
  data->fp_arr = (float **) malloc(sizeof(float *) * nlists);
  if(data->fp_arr == NULL){
    fprintf(stderr,"Unable to allocate memory\n");
    exit(1);
  }
  data->length = (long *) malloc(sizeof(long) * nlists);
  if(data->length == NULL){
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }
  for(int i = 0; i < nlists; i++)
    data->length[i] = load_array(data, names[i], i);
  return;
}
