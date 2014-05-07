#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common/io.h"
#include <time.h>
#include <cuda.h>

//#include "common/cuPrintf.cu"
/************************** DATA DEFINITIONS *******************************/
// CONSTANTS
#define THREADS_PER_BLOCK 64
#define M 2
#define OOOO 0x0
#define OOOl 1
#define OOlO 2
#define OOll 3
#define OlOO 4
#define OlOl 5
#define OllO 6
#define Olll 7
#define lOOO 8
#define lOOl 9
#define lOlO 10
#define lOll 11
#define llOO 12
#define llOl 13
#define lllO 14
#define llll 15

// MACROS
#define swap(A,B) { float temp = A; A = B; B = temp;}
#define compswap(A,B) if(B < A) swap(A,B)
#define digit(A,B) (((A) >> (8 - ((B)+1) * 8)) & ((1 << 8) - 1))
#define ch(A) digit(A, D)

// STRUCTS AND ENUM'S
typedef enum {
  BUCKET0 = 0,
  BUCKET1,
  BUCKET2,
  BUCKET3,
  BUCKET4,
  BUCKET5,
  BUCKET6,
  BUCKET7,
  BUCKET8,
  BUCKET9,
  BUCKET10,
  BUCKET11,
  BUCKET12,
  BUCKET13,
  BUCKET14,
  BUCKET15,
}state;

typedef struct data{
  int val;
  state bucket;
} data;

/**************************** CPU RADIX SORT *********************************/

/* insert
 *
 * Once the size of the list being sorted gets small enough, the overhead of
 * the radix quicksort implementation actually hampers performance, so we 
 * include an implementation of insertion sort to handle these short sublists.
 *
 * Parameters:
 * ls: The entire list of unsigned integers being sorted by Radix Quick Sort
 * l: The left bound of the section of ls being sorted by insert
 * r: The right bound of the section of ls being sorted by insert
 *
 */
void insert(int ls[], int l, int r){
  int i;
  for(i = r; i > l; i--) 
    compswap(ls[i-1], ls[i]);
  for(i = l + 2; i <= r; i++){
    int j = i;
    uint v = ls[i];
    while(v < ls[j-1]){
      ls[j] = ls[j-1];
      j--;
    }
    ls[j] = v;
  }
  return;
}

/* RadixQuicksort
 *
 * This is an implementation of the Radix Quicksort algorithm described in 
 * 'Algorithms in C' by Robert Sedgewick, Program 10.3 (page 422).
 *
 * Parameters:
 * ls: The list of unsigned integers being sorted.
 * l: The left bound of the section of ls being operated on in this call to
 *    RadixQuicksort.
 * r: The right bound of the section of ls being operated on in this call to
 *    RadixQuicksort.
 * D: The radix currently being compared, that is the index of the bit 
 *    (valued from 0 to 31) by which elements of ls are currently being 
 *    sorted by.
 */
void RadixQuicksort(int ls[], int l, int r, int D){
  int i, j, k, p, q, v;
  if(r-l <= M){
    insert(ls, l, r);
    return;
  }
  v = ch(ls[r]);
  i = l-1;
  j = r;
  p = l-1;
  q = r;
  while(i < j){
    while(ch(ls[++i]) < v);
    while (v < ch(ls[--j]))
      if(j == l)
	break;
    if(i > j)
      break;
    swap(ls[i],ls[j]);
    if(ch(ls[i]) == v){
      p++;
      swap(ls[p],ls[i]);
    }
    if(ch(ls[j]) == v){
      q--;
      swap(ls[j], ls[q]);
    }
  }
  if(p == q){
    if(v != '\0'){
      RadixQuicksort(ls, l, r, D+1);
      return;
    }
  }
  if(ch(ls[i]) < v)
    i++;
  for(k=l; k <= p; k++, j--)
    swap(ls[k], ls[j]);
  for(k=r; k >= q; k--, i++)
    swap(ls[k], ls[i]);
  RadixQuicksort(ls, l, j, D);
  if((i == r) && (ch(ls[i]) == v))
    i++;
  if( v != '\0' )
    RadixQuicksort(ls, j+1, i-1, D+1);
  RadixQuicksort(ls, i, r, D);
  return;
}

/* cpu_radixsort
 *
 * This is the wrapper function around RadixQuicksort, the purpose of which
 * is to set up the floating point array (convert it to integers),
 * set up the timing, call RadixQuicksort and then cast the integers
 * back into floating points.
 *
 * Parameters:
 * unsorted: The list of floating points to be sorted.
 * length: The length of the arrays
 * sorted: An output parameter, contains the list of floating points after the
 *         sorting algorithm has been executed.
 *
 * Return Value:
 * time: This function returns the time of execution of the sorting algorithm
 *       as a double precision floating point.
 */
double cpu_radixsort(float unsorted[], int length, float sorted[]){
  //1. Convert float * unsorted to int *
  int flipped[length];
  for(int i = 0; i< length; i++){
    flipped[i] = (int) (unsorted[i] * 1000000);
  }

  //2. Perform Radix Sort
  time_t start, end;
  double time;
  start = clock();

  //radix_sort call
  RadixQuicksort(flipped, 0, length - 1, 0);

  end = clock();
  time = ((double) end - start) / CLOCKS_PER_SEC;

  //3. Convert uint * to float *
  for(int i = 0; i < length; i++)
    sorted[i] = ((float) flipped[i]) / 1000000;

  return time;
}


/**************************** GPU RADIX SORT ********************************/

/* gpuRadixBitSort
 *
 * This kernel function implements the comparison and partition portion of an
 * implementation of Radix Quicksort.
 *
 * Parameters:
 * input: The input array of data structs to be sorted
 * output: An output parameter, will be populated with the partitioned/
 *         partially ordered list of data structs. As input, is identical
 *         to the input[] array, but the region [l,r] will be partitioned
 *         properly by the function.
 * l: The index of the left bound of the input being sorted.
 * r: The index of the right bound of the input being sorted.
 * nZeroes: An output parameter as well as counting struct. Each thread block
 *          keeps a counter of the number of values with a '0' at position D,
 *          this is used by the Host (after the function returns) to calculate
 *          where the partition between 0 and 1 occurs to recursively call
 *          gpuRadixBitSort.
 * nOnes: A counting struct. Similar in function to nZeroes, but for counting 
 *        '1' values in blocks.
 * D: The Radix, ranging from 0 to 31, this keeps track of which bit is being 
 *    compared.
 */
__global__ void gpuRadixBitSort(data input[], data output[], int l, int r, 
				int nZeroes[], int nOnes[], int D)
{
  __shared__ data bInput[THREADS_PER_BLOCK];

  if(threadIdx.x == 0){
    nZeroes[blockIdx.x] = 0;
    nOnes[blockIdx.x] = 0;
  }
  __syncthreads();

  int idx = l+blockIdx.x*THREADS_PER_BLOCK+threadIdx.x;

  if(idx <= r){
    bInput[threadIdx.x] = input[idx];
    int f = ((bInput[threadIdx.x].val) & (0x1111 << D)) >> D;
    switch (f){
      case OOOl:
	break;
      case OOlO:
	break;
      case OOll:
	break;
      case OlOO:
	break;
      case OlOl:
	break;
      case OllO:
	break;
      case Olll:
	break;
      case lOOO:
	break;
      case lOOl:
	break;
      case lOlO:
	break;
      case lOll:
	break;
      case llOO:
	break;
      case llOl:
	break;
      case lllO:
	break;
      case llll:
	break;
      default:
	bInput[threadIdx.x].bucket = BUCKET0;
	atomicAdd(&(nZeroes[blockIdx.x]), 1);
    }
    if( !f ){
    } else {
      bInput[threadIdx.x].bucket = BUCKET1;
      atomicAdd(&(nOnes[blockIdx.x]), 1);
    }
  }
  __syncthreads();

  if(threadIdx.x == 0){
    int lOffset = l;
    int rOffset = r;
    int i;
    for(i = 1; i <= blockIdx.x; i++){
      lOffset += nZeroes[i];
      rOffset -= nOnes[i];
    }
    int m = 0;
    int n = 0;
    for(int j = 0; j < THREADS_PER_BLOCK; j++){
      int chk = l + THREADS_PER_BLOCK*blockIdx.x + j;
      if( chk <= r){
	if(bInput[j].bucket == BUCKET0){
	  output[lOffset + m] = bInput[j];
	  ++m;
	} else {
	  output[rOffset - n] = bInput[j];
	  ++n;
	}
      }
    }
  }

  return;
}

/* grSort
 *
 * This function performs the recursive GPU Radix quick sorting by calculating 
 * the number of thread blocks requires, calling gpuRadixBitSort and 
 * recursively calling itself while ranging over the Radix value (0 to 31).
 *
 * Parameters:
 * ls: The list of data structs being sorted.
 * l: The left bound index on the section of ls being sorted currently.
 * r: The right bound index on the section of ls being sorted currently.
 * length: The length of ls.
 * D: The Radix, the index ranging from 0 to 31 corresponding to the bit being
 *    sorted at the moment.
 */
void grSort(data list[], int l, int r, int length, int D){
  if(D < 31){
    int numBlocks = (r - l) / THREADS_PER_BLOCK;
    if((numBlocks * THREADS_PER_BLOCK) < length)
      numBlocks++;

    data * d_list;
    data * d_list2;
    int * Zeroes;
    int * Ones;
    int divide[numBlocks];
    int size = sizeof(data);
    cudaMalloc(&d_list, size*length);
    cudaMalloc(&d_list2, size*length);
    cudaMalloc(&Zeroes, sizeof(int)*numBlocks);
    cudaMalloc(&Ones, sizeof(int)*numBlocks);
    cudaMemcpy(d_list, list, size*length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_list2, list, size*length, cudaMemcpyHostToDevice);
    
    gpuRadixBitSort<<<numBlocks, THREADS_PER_BLOCK>>>(d_list, d_list2, l, r, 
						      Zeroes, Ones, D);
    
    cudaMemcpy(list, d_list2, size*length, cudaMemcpyDeviceToHost);
    cudaMemcpy(&(divide), Zeroes, sizeof(int)*numBlocks, cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
    //cudaPrintfDisplay(stdout,true);

    int nZeroes = 0;
    for(int i = 0; i < numBlocks; i++){
      nZeroes += divide[i];
    }
    if(nZeroes > l)
      grSort(list, l, nZeroes - 1, length, D+1);
    if(nZeroes < r)
      grSort(list, nZeroes, r, length, D+1);
  }
  return;
}

/* gpu_radixsort
 *
 * This function is a wrapper around the call to grSort, meant to handle the
 * conversions to and from a floating point array to a data struct array, as
 * well as the timing functions to measure the speed of the actual sorting 
 * algorithm.
 *
 * Parameters:
 * unsorted: The list of floating point values to be sorted
 * length: The length of the arrays
 * sorted: An output parameter, will contain the results of applying the
 *         sorting algorithm.
 *
 * Return Value:
 * time: This function returns the the time of execution of the gpu radix
 *       sorting algorithm as a double-precision floating point.
 */
double gpu_radixsort(float unsorted[], int length, float sorted[]){
  time_t start, stop;
  double time;

  data list[length];
  for(int i = 0; i< length; i++){
    list[i].val = (int) (unsorted[i] * 1000000);
    list[i].bucket = BUCKET0;
  }

  start = clock();
  grSort(list, 0, length - 1, length, 0);
  stop = clock();
  time = ((double) stop - start) / CLOCKS_PER_SEC;

  for(int j = 0; j < length; j++)
    sorted[j] = ((float) list[j].val) / 1000000;

  return time;
}

/* radixsort
 * 
 * This function makes calls to the CPU and GPU implementations of Radix Sort
 * and populates a Result Struct. It also performs a quick check to ensure
 * that the results of each sorting algorithm are consistent (a debugging
 * feature).
 *
 * Parameters:
 * unsorted: A list of floating points to be sorted
 * length: The length of the unsorted array
 * result: An output parameter to be populated with the name of the test and
 *         the times of execution of the CPU and GPU implementations of Radix
 *         sort.
 */
void radixsort(float unsorted[], int length, Result * result){

  //cudaPrintfInit();

  result = (Result *) malloc(sizeof(Result));
  if(result == NULL){
    fprintf(stderr, "Out of Memory\n");
    exit(1);
  }
  strcpy(result->tname, "Radix Sort");  
  float sorted[2][length];

  result->cpu_time = cpu_radixsort(unsorted, length, sorted[0]);
  result->gpu_time = gpu_radixsort(unsorted, length, sorted[1]);

  cudaThreadSynchronize();
  //cudaPrintfDisplay(stdout,true);
  //cudaPrintfEnd();

  //check that sorted[0] = sorted[1];
  int n = 0;
  for(int i = 0; i < length; i++){
    if(sorted[0][i] != sorted[1][i])
      n++;
    printf("[%d] CPU: %f\t GPU: %f\n", i, sorted[0][i], sorted[1][i]);
  }
  if(n!= 0){
    fprintf(stderr, "There were %d discrepencies between the CPU and GPU Radix Sort algorithms\n", n);
  }

  return;

}
