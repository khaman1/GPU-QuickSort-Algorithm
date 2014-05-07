#include <stdlib.h>
#include <stdio.h>
#include "io.h"

void print_data (Data * data, int elt_flag){
  printf("Number of Lists Read In: %d\n", data->nlists);
  for(int i = 0; i < data->nlists; i++){
    printf("List %d has %ld elements:\n", i, data->length[i]);
    if(elt_flag){
      for(int j = 0; j < data->length[i]; j++){
	if(!(j % 10))
	  printf("%f \n", data->fp_arr[i][j]);
	else
	  printf("%f \t", data->fp_arr[i][j]);
      }
      printf("\n");
    }
  }
}
