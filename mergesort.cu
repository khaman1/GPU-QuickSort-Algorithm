#include <math.h>
#include <stdio.h>

#define SHAREDSIZE 8000  /* Should be changed to dynamically detect shared
                             memory size if at all possible.  */

//Forward declarations
__global__ void GPUMerge(float *d_list, int len, int stepSize,
                         int eltsPerThread);

/* Mergesort definition.  Takes a pointer to a list of floats.
 * the length of the list, the number of threads per block, and 
 * the number of blocks on which to execute.  
 * Puts the list into sorted order in-place.*/
void MergeSort(float *h_list, int len, int threadsPerBlock, int blocks) {

    float *d_list;
    if ( (cudaMalloc((void **) &d_list, len*sizeof(float))) == 
         cudaErrorMemoryAllocation) {
        printf("Error:  Insufficient device memory at line %d\n", __LINE__);
        return;
    }

    cudaMemcpy(d_list, h_list, len*sizeof(float), cudaMemcpyHostToDevice);

    int stepSize = ceil(len/float(threadsPerBlock*blocks));
    int eltsPerThread = ceil(stepSize/threadsPerBlock);
    int maxStep = SHAREDSIZE/sizeof(float);

    if (maxStep < stepSize) {
        stepSize = maxStep;
    }

    GPUMerge<<<blocks, threadsPerBlock>>>(d_list, len, stepSize,
                                          eltsPerThread);

    cudaMemcpy(h_list, d_list, len*sizeof(float), cudaMemcpyDeviceToHost);

}

/* Mergesort definition.  Takes a pointer to a list of floats, the length
 * of the list, and the number of list elements given to each thread.
 * Puts the list into sorted order in-place.*/
__global__ void GPUMerge(float *d_list, int len, int stepSize,
                         int eltsPerThread){

    int my_start, my_end; //indices of each thread's start/end

    //Declare counters requierd for recursive mergesort
    int l_start, r_start; //Start index of the two lists being merged
    int old_l_start;
    int l_end, r_end; //End index of the two lists being merged
    int headLoc; //current location of the write head on the newList
    short curList = 0; /* Will be used to determine which of two lists is the
                        * most up-to-date, since merge sort is not an in-place
                        * sorting algorithm. */

    //Attempt to allocate enough shared memory for this block's list...
    //Note that mergesort is not an in-place sort, so we need double memory.
    __shared__ float subList[2][SHAREDSIZE/sizeof(float)];

    //Load memory
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = 0; i < eltsPerThread; i++){
        if (index + i < len){
            subList[curList][eltsPerThread*threadIdx.x + i]=d_list[index + i];
        }
    }

    //Wait until all memory has been loaded
    __syncthreads();
    
    //Merge the left and right lists.
    for (int walkLen = 1; walkLen < len; walkLen *= 2) { 
        //Set up start and end indices.
        my_start = eltsPerThread*threadIdx.x;
        my_end = my_start + eltsPerThread;
        l_start = my_start;

        while (l_start < my_end) { 
            old_l_start = l_start; //l_start will be getting incremented soon.
            //If this happens, we are done.
            if (l_start > my_end){
                l_start = len;
                break;
            }
            
            l_end = l_start + walkLen;
            if (l_end > my_end) {
                l_end = len;
            }
            
            r_start = l_end;
            if (r_start > my_end) {
                r_end = len;
            }
            
            r_end = r_start + walkLen;
            if (r_end > my_end) {
                r_end = len;
            }
            
            for (int i = 0; i < walkLen; i++){
                if (subList[curList][l_start] < subList[curList][r_start]) {
                    subList[!curList][headLoc] = subList[curList][l_start];
                    l_start++;
                    headLoc++;
                    //Check if l is now empty
                    if (l_start == l_end) {
                        for (int j = r_start; j < r_end; j++){
                            subList[!curList][headLoc] = 
                                subList[curList][r_start];
                            r_start++;
                            headLoc++;
                    }
                    } 
                }
                else {
                    subList[!curList][headLoc] = subList[curList][r_start];
                    r_start++;
                    //Check if r is now empty
                    if (r_start == r_end) {
                        for (int j = l_start; j < l_end; j++){
                            subList[!curList][headLoc] = 
                                subList[curList][r_start];
                            r_start++;
                            headLoc++;
                        }
                    } 
                }
            }

            l_start = old_l_start + 2*walkLen;
            curList = !curList;
        }
    }
    
    return;

    //subList[blockIdx

    //...otherwise, we use global memory...
    /*
    if ( (subList = cudaMalloc(stepsize*sizeof(float)) != NULL ) {
            //   do some shit.
            
        }    
    */

    //...otherwise, we give up.

}

int main(int argc, char *argv[]){
    
    int len;
    float *h_list;

    if (argc != 2) {
        printf("Invalid argument count.  %s requires exactly 1 argument, \
%d given",
               argv[0], argc);
        return -1;
    }
    
    FILE *fin = fopen(argv[1], "r");
    
    if (fin == NULL){
        printf("Could not open file: %s", argv[1]);
        return -2;
    }

    fscanf(fin, "%d", &len);

    h_list = (float *)malloc(len*sizeof(float));
    if (h_list == NULL){
        printf("Insufficient host memory to allocate at %d", __LINE__);
        return -3;
    }

    for (int i = 0; i < len; i++){
        fscanf(fin, "%f ", &h_list[i]);
    }

    for (int i = 0; i < len; i++){
        printf("%f\n", h_list[i]);
    }

    MergeSort(h_list, len, 8, 512);

    return 0;
}
