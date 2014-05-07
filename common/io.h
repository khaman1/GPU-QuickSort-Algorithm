#ifndef _IO_H_
#define _IO_H_

//++++++++++++++++++++++++++++ DATA DEFINITIONS +++++++++++++++++++++++++++++//

/* Data
 *
 * This struct contains all of the floating point data to be operated on
 * by the sorting functions.
 *
 * Contents:
 * nlists - This is how many lists are stored in the Data struct
 * length - This is an array of the lengths of each list of floating
 *          points stored in the Data struct, its size is determined by
 *          'nlists'
 * fp_arr - This is a 2D array containing all of the lists of floating
 *          point values to be operated on with the sorting algorithms.
 *          Its size is (nlists)x(max(length)). Length[i] provides the
 *          length (number of fp's) of the list fp_arr[i].
 */
typedef struct fp_array{
  int nlists;
  long * length; 
  float ** fp_arr; 
} Data;

/* Result
 *
 * This struct is contains the results of a sorting algorithm acting on a
 * single floating point list.
 *
 * Contents:
 * tname -    The name of the 'test,' that is the sorting algorithm for which
 *            these results correspond.
 * cpu_time - The amount of time required to complete the sorting algorithm
 *            on the cpu.
 * gpu_time - The amount of time required to compelte the sorting algorithm
 *            on the gpu.
 */
typedef struct result{
  char tname[64]; 
  double cpu_time;
  double gpu_time;
} Result;

/* Results
 *
 * This struct, as its name suggests, collate the Result structs into a 2D
 * array, accessible by test and list index values.
 *
 * Contents:
 * ntests - The number of sorting algorithms implemented.
 * nlists - The number of lists of floating points to be sorted by said tests.
 * rlist  - A 2D array of Result structs, indexed using test and list id's.
 */
typedef struct results{
  int ntests;
  int nlists;
  Result ** rlist;
} Results;

// +++++++++++++++++++++++++++ PARSING FUNCTIONS +++++++++++++++++++++++++++ //

/* Parser
 *
 * This function reads properly formatted list files into memory to use
 * in the sorting functions.
 *
 * Parameters:
 * data   - A Data struct to be populated.
 * nlists - The number of list files to be read into memory. These 
 *          files must match the format from being created using
 *          'generatefp.c'
 * names  - The file names passed as arguments on the 'main' function
 *          call.
 *
 * Return Values:
 * void
 */
void parser(Data * data, int nlists, char ** names);


// ++++++++++++++++++++++++++++ DEBUG FUNCTIONS ++++++++++++++++++++++++++++ //

/* Print_Data
 * 
 * This function prints the contents of a Data struct
 *
 * Parameters:
 * data     - The Data Struct to be printed
 * elt_flag - This int is treated as a bool, if 1 print all the elements of
 *            every list, otherwise only print the meta data of the lists.
 *
 * Return Values:
 * void 
 */
void print_data (Data * data, int elt_flag);

// ++++++++++++++++++++++++++++ SORTING ALGORITHMS +++++++++++++++++++++++++ //

void quicksort(float unsorted[], int length, Result * result);

void radixsort(float unsorted[], int length, Result * result);

#endif
