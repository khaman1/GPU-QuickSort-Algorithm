#include <math.h>
#include <stdio.h>
#include "common/cuPrintf.cu"

/* Heapsort.cu:  Individual block sorting works at this point, and is stored
 * in the "midlist" structure passed to the GPUheapsort function.  Block
 * combinations via metaheaps do not work.
 */

#define BLOCKSIZE 1023 //Size of blocks at the bottom heap
#define METASIZE 511   //Size of metaheap
#define METACACHE 4    //Size of metacache
#define METADEPTH 9    //Max depth of metaheap (ceil of log of metaSize)
#define OUTSIZE 512      //Size of output shared memory
#define BLOCKDEPTH 10  //Max Depth of bottom heap, and ceil of log of blocksize
#define MINWARPS 1     //Minimum warp count to run code.
#define INVALID -1     //special value signifying invalid Buffer/heap entry.

typedef struct metaEntry {
    float value;
    short key;
} metaEntry_t;


//Tells us our current progress on building a given block.
typedef struct blockInfo{
    int bufSize;      //How many popped elements are buffered right now?
    int writeLoc;     //Index into blockwrite array for next write
    int readLoc;      //Index into blockwrite array for next read
    short remaining;  //How many elements are left to pop?
    short size;       //Total number of elements in the block.  0 iff uninit.
    short index;      //Which block are we?
    short heapified;  //Only a bool is needed, but short maintains alignment.
} blockInfo_t;


//Forward declarations
__global__ void GPUHeapSort(float *d_list, float *midList, float *sortedList,
                            blockInfo_t *blockInfo,
                            int numBlocks, int len, int topHeapSize,
                            int botHeapSize, 
                            int warpSize, int metaDepth);
__device__ void bottomLevel(float *d_list, int len); //NYI
__device__ void topLevel(float *d_list, int len); //NYI
__device__ void heapify(__volatile__ float *in_list, int len);
__device__ void metaHeapify(__volatile__ metaEntry_t *in_list,
                            __volatile__ float buf[METASIZE][METACACHE], int len);
__device__ void pipelinedPop(__volatile__ float *heap, float *out_list, 
                             int d, int popCount);
__device__ void fillBuffer(float *g_block, blockInfo_t *blockInfo,
                           float *buffer, int firstThread, int *isDone, 
                           int nextBlock_temp);
__device__ void loadBlock(float *g_block, float *s_block,
                          blockInfo_t *g_info, blockInfo_t *s_info);
__device__ void writeBlock(float *g_block, float *s_block,
                           int writeLen,
                           blockInfo_t *g_info, blockInfo_t *s_info);
__device__ void printBlock(float *s_block, int blockLen);
__device__ void initBlocks(blockInfo_t *blockInfo, int numBlocks, int len);
__device__ void initMetaHeap(metaEntry *heap, float buf[METASIZE][METACACHE]);
__host__ int heapSort(float *h_list, metaEntry_t *superTemp,
                      int len, int threadsPerBlock,
                      int blocks, cudaDeviceProp devProp);
__host__ int floorlog2(int x);

//Ceiling of log2 of x.  Could be made faster, but effect would be negligible.
int ceilLog2(int x){
    if (x < 1){
        return -1;
    }
    x--;
    int output = 0;
    while (x > 0) {
        x >>= 1;
        output++;
    }
    return output;
}

/* Heapsort definition.  Takes a pointer to a list of floats.
 * the length of the list, the number of threads per block, and 
 * the number of blocks on which to execute.  
 * Puts the list into sorted order in-place.*/
int heapSort(float *h_list, metaEntry_t *superTemp,
             int len, int threadsPerBlock, int blocks,
              cudaDeviceProp devProp) {

    float *d_list, *midList, *sortedList; //various lists that will live on GPU
    blockInfo_t *blockInfo;
    blockInfo_t *dummyBlocks;  //A bunch of zeroed blockinfos to zero GPU mem.
    int logLen; //log of length of list
    int metaDepth; //layers of metaheaps
    int topHeapSize; //Size of the top heap
    int logBotHeapSize; //log_2 of max size of the bottom heaps 
    int logMidHeapSize; //log_2 of max size of intermediate heaps
    int numBlocks; //Number of bottom heaps.  Poor choice of name =p.
    int temp;

    //Trivial list?  Just return.
    if (len < 2){
        return 0;
    }

    //Ensure that we have a valid number of threads per block.
    if (threadsPerBlock == 0){
        threadsPerBlock = devProp.maxThreadsPerBlock;
    }
    //We require a minimum of 2 warps per block to run our code
    else if (threadsPerBlock < 2*devProp.warpSize){
        printf("At least 2 warps are required to run heapsort.  ");
        printf("Increasing thread count to 64.\n");
        threadsPerBlock = MINWARPS*devProp.warpSize;
    }
    if (threadsPerBlock > devProp.maxThreadsPerBlock) {
        printf("Device cannot handle %d threads per block.  Max is %d\n",
               threadsPerBlock, devProp.maxThreadsPerBlock);
        return -1;
    }
    //We require a minimum of 2 blocks to run our code.
    if (blocks < 2){
        printf("At least 2 blocks are required to run heapsort.\n");
        return -1;
    }
     
    //Calculate size of heaps.  BotHeapSize is 1/8 shared mem size.
    //logBotHeapSize = ceilLog2(devProp.sharedMemPerBlock>>3);
    logBotHeapSize = BLOCKDEPTH;
    logMidHeapSize = logBotHeapSize - 2;

    printf("logBotHeap: %d, logMidHeap: %d\n", logBotHeapSize, logMidHeapSize);

    //Calculate metaDepth and topHeapSize.
    metaDepth = 0; //Will increment this if necessary.
    logLen = ceilLog2(len);
    temp = logBotHeapSize; //temp is a counter tracking total subheap depth.
    
    //Do we only need one heap?
    if (temp >= logLen){
        topHeapSize = len;
    }
    //Otherwise, how many metaheaps do we need?
    else {
        while (temp < logLen){
            metaDepth++;
            temp += logMidHeapSize;
        }
        topHeapSize = len>>temp;
    }

    //Nevermind the fancy calculations above... let's just do this.
    topHeapSize = ceil((float)len/BLOCKSIZE);

    printf("metaDepth is %d\n", metaDepth);
    printf("len is %d, blocksize is %d\n", len, BLOCKSIZE); 
    printf("topHeapSize is %d\n", topHeapSize); 

    if (metaDepth > blocks){
        printf("Must have at least metaDepth blocks available.");
        printf("metaDepth is %d, but only %d blocks were given.\n", 
               metaDepth, blocks);
        return -1;
    }

    if (metaDepth > 2){
        printf("Current implementation only supports metaDepth of 2.  ");
        printf("Given metadepth was %d.  In practice, ", metaDepth); 
        printf("this means that list lengths cannot equal or exceed 2^20.");
    }


    if ( (cudaMalloc((void **) &d_list, len*sizeof(float))) == 
         cudaErrorMemoryAllocation) {
        printf("Error:  Insufficient device memory at line %d\n", __LINE__);
        return -1;
    }

    if ( (cudaMalloc((void **) &midList, len*sizeof(float))) == 
         cudaErrorMemoryAllocation) {
        printf("Error:  Insufficient device memory at line %d\n", __LINE__);
        return -1;
    }

    if ( (cudaMalloc((void **) &sortedList, len*sizeof(float))) == 
         cudaErrorMemoryAllocation) {
        printf("Error:  Insufficient device memory at line %d\n", __LINE__);
        return -1;
    }

    cudaMemcpy(d_list, h_list, len*sizeof(float), cudaMemcpyHostToDevice);

    numBlocks = ceil((float)len/BLOCKSIZE); //number of bottom heaps
    printf("numHBlocks: %d\n", numBlocks);

    dummyBlocks = (blockInfo_t *)calloc(numBlocks, sizeof(blockInfo_t));


    if ( (cudaMalloc((void **) &blockInfo, len*sizeof(blockInfo_t))) == 
         cudaErrorMemoryAllocation) {
        printf("Error:  Insufficient device memory at line %d\n", __LINE__);
        return -1;
    }

    cudaMemcpy(blockInfo, dummyBlocks, len*sizeof(float),
               cudaMemcpyHostToDevice);

    printf("Attempting to call GPUHeapSort\n\n");

    GPUHeapSort<<<blocks, threadsPerBlock>>>
        (d_list, midList, sortedList, blockInfo, numBlocks, 
         len, topHeapSize, BLOCKSIZE, devProp.warpSize, metaDepth);

    cudaThreadSynchronize();
    
    cudaMemcpy(h_list, midList, len*sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/* GPUHeapSort definition.  Takes a pointer to a list of floats, the length
 * of the list, and the number of list elements given to each thread.
 * Puts the list into sorted order in-place.*/
__global__ void GPUHeapSort(float *d_list, float *midList, float *sortedList,
                            blockInfo_t *blockInfo,
                            int numBlocks, int len, int topHeapSize,
                            int botHeapSize,
                            int warpSize, int metaDepth){
    

    __shared__ float heap[BLOCKSIZE];
    __shared__ float buffer[METASIZE][METACACHE];
    __shared__ float output[OUTSIZE];
   

    if (blockIdx.x == 0) { 

        __shared__ int isDone;  //Tracks how many blocks we have loaded.
        __shared__ metaEntry_t *metaHeap;
        int nextBlock;
        

        if (threadIdx.x == 32){
            metaHeap = (metaEntry_t *)heap; //reuse the "heap" declaration.
            isDone = 0;
        }

        __syncthreads();
        //cuPrintf("numBlocks reaches %d just before initing\n", numBlocks);
        //Initialize datastructures
        initMetaHeap(metaHeap, buffer);
        //cuPrintf("numBlocks eez %d\n", numBlocks);
        __syncthreads();

        //First warp maintains metaheap.
        if (threadIdx.x < 31){
            //cuPrintf("topHeapSize is %d\n", topHeapSize);
            metaHeapify(metaHeap, buffer, topHeapSize);
            //cuPrintf("warp 0 says that isDone is now %d\n", isDone);
        }
        //Laters warps maintain buffers.
        else {
            nextBlock = (threadIdx.x>>5) -1;
            while (isDone < numBlocks) {
                //cuPrintf("isDone is %d\n", isDone);
                
                fillBuffer(&midList[nextBlock*BLOCKSIZE],
                           &blockInfo[nextBlock], buffer[nextBlock],
                           threadIdx.x & ~31, &isDone, nextBlock);
                
                nextBlock += (blockDim.x>>5)-1;
                if (nextBlock > numBlocks){
                    nextBlock = (threadIdx.x>>5)-1;
                    }
            }
        }        
        __syncthreads();

        //Temporary -- write the metaheap to memory.
        for (int i = threadIdx.x; i < topHeapSize; i+= blockDim.x){
            sortedList[i] = metaHeap[i].value;
        }
        __syncthreads();


    }
    else {
        __shared__ blockInfo_t curBlockInfo;
        __shared__ int popCount; //How many heap elements are we popping?
        int curIdx; //The index of the current block
 

        //cuPrintf("About to call init\n");
        
        //Initialize datastructures
        initBlocks(blockInfo, numBlocks, len);
        //cuPrintf("Init finished.\n");
        __syncthreads();
        
        curIdx = blockIdx.x-1;
        
        while (curIdx < numBlocks){
            //Load memory
            loadBlock(&d_list[curIdx*BLOCKSIZE], (float *)heap,
                      &blockInfo[curIdx], &curBlockInfo);
            
            /*cuPrintf("curBlockInfo:  (bufsize: %d, writeloc: %d, heapified: %d \
 remaining: %d, size: %d\n",
                     curBlockInfo.bufSize, curBlockInfo.writeLoc,
                     curBlockInfo.heapified, curBlockInfo.remaining,
                     curBlockInfo.size);
            */ 
            if (curBlockInfo.heapified == 0){
                //First warp heapifies
                if (threadIdx.x < 8){
                    heapify(heap, curBlockInfo.size);
                }
                curBlockInfo.heapified = 1;
                __syncthreads();
            }
            //cuPrintf("Entering while Loop\n");
            while (curBlockInfo.remaining > 0){
                //First warp pops
                
                /*cuPrintf("curBlockInfo:  (bufsize: %d, writeloc: %d, heapified: %d \
 remaining: %d, size: %d\n",
                         curBlockInfo.bufSize, curBlockInfo.writeLoc,
                         curBlockInfo.heapified, curBlockInfo.remaining,
                         curBlockInfo.size);
                */
                
                if (threadIdx.x == 0){
                    popCount = curBlockInfo.remaining;
                    if (popCount > OUTSIZE) {
                        popCount = OUTSIZE;
                    }
                    curBlockInfo.remaining -= popCount;
                }
                if (threadIdx.x < 8){
                    pipelinedPop(heap, (float *)output, BLOCKDEPTH, popCount);
                }
                
                __syncthreads();
                //cuPrintf("Calling writeBlock with popcount %d\n", popCount);
                
                writeBlock(midList, output, popCount,
                           &blockInfo[curBlockInfo.index], &curBlockInfo);
                __syncthreads();
                //cuPrintf("At end, remaining:  %d\n", curBlockInfo.remaining);
            }
            //cuPrintf("After the while loop...\n");
        
        curIdx += (gridDim.x - 1);
        }
    }
    return;

}

/* Loads a block of data from global memory into shared memory.  Must be
 * called by all threads of a thread block to ensure proper operation.
 * g_info:  A pointer to the specific (global) blockinfo to be read. 
 */
__device__ void loadBlock(float *g_block, float *s_block,
                          blockInfo_t *g_info, blockInfo_t *s_info){
    
    if (threadIdx.x == 0){
        *s_info = *g_info;
        __threadfence_block();
    }

    //cuPrintf("Entering loadBlock\n");
    for(int i = threadIdx.x; i < BLOCKSIZE; i += gridDim.x){
        if(i < s_info->size){
            s_block[i] = g_block[i];
        } 
        else {
            s_block[i] = 0;
        }
    }
    __syncthreads();
    return;
}

/*
 * fillBuffer:  Fills empty slots in buffer with data from g_block.
 *              Expects to be called by METACACHE blocks.
 * g_block:  Pointer to the location we should start reading from.
 * blockInfo:  Pointer to the blockInfo struct for the g_block.
 * buffer:  pointer to the location of a float[4] buffer for the metaheap.
 * firstThread:  The first of METACACHE contiguous threads running this func.
 */
__device__ void fillBuffer(float *g_block, blockInfo_t *blockInfo,
                           float *buffer,
                           int firstThread, int *isDone,
                           int nextBlock_temp){
    __shared__ int readLoc;
    __shared__ int isReady;
    __shared__ int writeLoc;
    __shared__ int toCacheCount; //How many new elements do we need to cache?
    int index;

    //The first three threads perform slow memory operations.  The fourth
    //does some processing while the others are waiting for memory.
    if(threadIdx.x == firstThread) {
        isReady = blockInfo->writeLoc;
        cuPrintf("Blocksize begins as %d\n", blockInfo->size);
    }
    if (threadIdx.x == firstThread+1){
        readLoc = blockInfo->readLoc;
    }
    if (threadIdx.x == firstThread+2){
        writeLoc = blockInfo->writeLoc;
    }
    if (threadIdx.x == firstThread+3){
        toCacheCount = 0;
        for (int i = 0; i < METACACHE; i++){
            if (buffer[i] == INVALID) {
                toCacheCount++;
            }
        }
    }
    __threadfence_block();

    //Block isn't ready.  Do nothing.
    if (isReady == 0){
        //cuPrintf("Not ready yet!\n");
        return;
    }
    else {
        //cuPrintf("Totally ready!\n");
        //For threads firstThread through METACACHE...
        index = threadIdx.x - firstThread;
        if (index < METACACHE){
            //cuPrintf("index < METACACHE! ! !\n");
            //Note that the following three lines risk a race w/warp 0.  Alas, 
            //atomic TAS on shared is not implemented in compute capacity 1.1, so this is
            //more-or-less unavoidable.  We need to atomically test the buffer for 
            //INVALID state and then write to it to avoid the race condition.  If this
            //were going to be run on something of compute capacity 1.2 or greater, 
            //an atomic TAS would be used and all would be well.
            //Do we have an invalid cache entry?
            if (buffer[index] == INVALID) {
                //cuPrintf("INVALID index! ! !\n");
                //Does g_block have an element that can fill our invalid entry?
                if (readLoc + index < writeLoc){                     
                    buffer[index] = g_block[readLoc+index];
                    /*cuPrintf("filled %d th buffer %d with %f\n", 
                             nextBlock_temp, index,  buffer[index]);
                    cuPrintf("buffer addr is %d\n", (long)&buffer[index]);
                    cuPrintf("read addr is %d\n",
                    (long)&g_block[readLoc+index]);*/
                }
            }
        }
    }

    if (threadIdx.x == firstThread){
        if (readLoc + toCacheCount > writeLoc){
            toCacheCount = writeLoc - readLoc;
            __threadfence_block();
        }
    }

    //update the blockData structure in global memory
    if (threadIdx.x == firstThread){
        atomicSub(&blockInfo->bufSize, toCacheCount);  
    }
    if (threadIdx.x == firstThread + 1){
        //cuPrintf("ended up caching %d elements\n", toCacheCount);
        blockInfo->readLoc = readLoc + toCacheCount;
    }
    if (threadIdx.x == firstThread + 2){
        if (toCacheCount > 0){
            *isDone = *isDone + 1;
        }
    }
    return;
}

/* Writes a block of data from shared memory into global memory.  Must be
 * called by all threads of a thread block to ensure proper operation. 
 * g_info:  A pointer to the specific (global) g_info to be written.
 */
__device__ void writeBlock(float *g_block, float *s_block, int writeLen,
                           blockInfo_t *g_info, blockInfo_t *s_info){
    //cuPrintf("beginning writeBlock\n");
    for(int i = threadIdx.x; i < writeLen; i += blockDim.x){
        g_block[s_info->writeLoc+i] = s_block[i];
        //cuPrintf("writing block %d to %d with value %f\n",
        //         i, s_info->writeLoc + i, g_block[s_info->writeLoc + i]);
    }
    __syncthreads();
    //Update the blockInfo struct in global memory
    if (threadIdx.x == 0){
        //s_info->writeLoc += writeLen;
        //*g_info = *s_info;
        //cuPrintf("setting writeLoc to %d\n", s_info->writeLoc + writeLen);
        s_info->writeLoc += writeLen;
        g_info->writeLoc = s_info->writeLoc;
        g_info->remaining = s_info->remaining;
        atomicAdd(&g_info->bufSize, writeLen);
    }
    __syncthreads();
    return;
}

/* Writes a chunk of raw floats from shared memory into global memory.  Must be
 * called by all threads of a thread block to ensure proper operation. 
 * g_info:  A pointer to the specific (global) g_info to be written.
 */
__device__ void writeRawFloats(float *g_block, float *s_block, int writeLen){
                             
    //cuPrintf("beginning writeBlock\n");
    for(int i = threadIdx.x; i < writeLen; i += blockDim.x){
        g_block[i] = s_block[i];
        //cuPrintf("writing block %d to %d with value %f\n",
        //         i, s_info->writeLoc + i, g_block[s_info->writeLoc + i]);
    }
    __syncthreads();
    return;
}

/* Prints a block of data in shared memory */
__device__ void printBlock(float *s_block, int blockLen){
    for (int i = threadIdx.x; i < blockLen; i += blockDim.x){
        cuPrintf("s_block[%d] = %f\n", i, s_block[i]);
    }
}

/* Initializes data structures for heapsort.  Must be run by all threads
 * of all nonzero blocks.
 * blockInfo:  A pointer to the entire array of blockInfos.
 */
__device__ void initBlocks(blockInfo_t *blockInfo, int numBlocks, int len){

    if ((threadIdx.x == 0) && (blockIdx.x != 0) ){

        cuPrintf("attempting to init\n");
        //Initialize blockinfo structs.  Initialization is done by the blocks
        //that own each blockinfo struct.
        blockInfo_t BI;
        BI.bufSize = 0;
        BI.heapified = 0;
        BI.readLoc = 0;
        BI.remaining = BLOCKSIZE;
        BI.size = BLOCKSIZE;
        for (int idx = (blockIdx.x-1); idx < numBlocks; idx += (gridDim.x-1)){
            BI.writeLoc = idx*BLOCKSIZE;
            BI.index = idx;
            cuPrintf("writeloc is %d\n", BI.writeLoc);
            //Did we overrun our bounds when setting size?
            if ((idx+1)*BLOCKSIZE > len){
                BI.size = len - idx*BLOCKSIZE;
                BI.remaining = BI.size;
            }
            blockInfo[idx] = BI;
        }
    }
    __syncthreads();
}

__device__ void initMetaHeap(metaEntry *heap, float buf[METASIZE][METACACHE]){

    for (int i = threadIdx.x; i < METASIZE; i += blockDim.x){
        heap[i].value = INVALID;
        heap[i].key = i;
        for (int j = 0; j < METACACHE; j++){
            buf[i][j] = INVALID;
            //cuPrintf("Invalidating buf[%d][%d]\n", i, j); 
        }
    }

    return;
}

/* Heapifies a list using a single warp.  Must be run on the bottom warp of a
 * thread.  If this function is not executed by all of threads 0-7, the GPU
 * will stall.
 */
__device__ void heapify(__volatile__ float *inList, int len){
    
    int focusIdx = 0; //Index of element currently being heapified
    float focus=0, parent=0; //current element being heapified and its parent
    __volatile__ __shared__ int temp;
    /*int localTemp=0; Temp doesn't need to be re-read _every_ time.
                    * Temp will be used to track the next element to percolate.
                    */

    if (threadIdx.x == 0){
        temp = 0; //Index of next element to heapify
    }

    //localTemp = 0;
    
    //We maintain the invariant that no two threads are processing on
    //adjacent layers of the heap in order to avoid memory conflicts and
    //race conditions.
    while (temp < len){
        if (threadIdx.x == (temp & 7)){
            focusIdx = temp;
            focus = inList[focusIdx];
            temp = temp + 1;
            //cuPrintf("Focusing on element %d with value %f\n",
            //         focusIdx, focus);
        }
        
        //Unrolled loop once to avoid race conditions and get a small speed
        //boost over using a for loop on 2 iterations.
        if (focusIdx != 0){
            parent = inList[(focusIdx-1)>>1];
            //Swap focus and parent if focus is bigger than parent
            if (focus > parent){
                //cuPrintf("Focus %f > parent %f\n", focus, parent); 
                inList[focusIdx] = parent;
                inList[(focusIdx-1)>>1] = focus;
                focusIdx = (focusIdx - 1)>>1;
            }
            else {
                //cuPrintf("Parent %f > focus %f\n", parent, focus);
                focusIdx = 0;
            }
        }
        if (focusIdx != 0){
            parent = inList[(focusIdx-1)>>1];
            //Swap focus and parent if focus is bigger than parent
            if (focus > parent){
                //cuPrintf("Focus %f > parent %f\n", focus, parent); 
                inList[focusIdx] = parent;
                inList[(focusIdx-1)>>1] = focus;
                focusIdx = (focusIdx-1)>>1;
            }
            else {
                //cuPrintf("Parent %f > focus %f\n", parent, focus);
                focusIdx = 0; 
            }
       }
        //localTemp = *temp;
    }
    
    //Empty the pipeline before returning
    while (focusIdx !=0){
        parent = inList[(focusIdx-1)>>1];
        //Swap focus and parent if focus is bigger than parent
        if (focus > parent){
            cuPrintf("Focus %f > parent %f\n", focus, parent); 
            inList[focusIdx] = parent;
            inList[(focusIdx-1)>>1] = focus;
            focusIdx = (focusIdx-1)>>1;
        }
        else {
            //cuPrintf("Parent %f > focus %f\n", parent, focus);
            focusIdx = 0; 
        }
    }
    
    return;
}

/* Heapifies a list using a single warp.  Must be run on the bottom warp of a
 * thread.  If this function is not executed by all of threads 0-7, the GPU
 * will stall.
 */
__device__ void metaHeapify(__volatile__ metaEntry_t *inList,
                            __volatile__ float buf[METASIZE][METACACHE], 
                            int len){
    
    int focusIdx = 0; //Index of element currently being heapified
    __volatile__ metaEntry_t focus, parent; //current element being heapified and its parent
    __volatile__ __shared__ int temp;
    /*int localTemp=0; Temp doesn't need to be re-read _every_ time.
                    * Temp will be used to track the next element to percolate.
                    */

    if (threadIdx.x == 0){
        temp = 0; //Index of next element to heapify
    }

    //localTemp = 0;
    
    //We maintain the invariant that no two threads are processing on
    //adjacent layers of the heap in order to avoid memory conflicts and
    //race conditions.
    while (temp < len){
        if (threadIdx.x == (temp & 7)){
            focusIdx = temp;
            focus.key = inList[focusIdx].key;
            focus.value = inList[focusIdx].value;
            //Initialize the inList entry
            do {
                inList[focusIdx].value = buf[focus.key][0];
                focus.value = inList[focusIdx].value;
                /*
                if ((i % 1000) == 0) {
                    cuPrintf("(key, value) is (%d, %f)\n",
                             focus.key, focus.value);
                }
                i++;
                */
                //} while (i < 10000);
            } while (focus.value == INVALID);
            temp = temp + 1;
            /*cuPrintf("Focusing on element %d with value %f\n",
              focusIdx, focus.value);*/
        }
        
        //Unrolled loop once to avoid race conditions and get a small speed
        //boost over using a for loop on 2 iterations.
        if (focusIdx != 0){
            parent.key = inList[(focusIdx-1)>>1].key;
            parent.value = inList[(focusIdx-1)>>1].value;
            //Swap focus and parent if focus is bigger than parent
            if (focus.value > parent.value){
                cuPrintf("Focus %f > parent %f.  Setting index %d to focus\n",
                         focus.value, parent.value, (focusIdx-1)>>1); 
                inList[focusIdx].key = parent.key;
                inList[focusIdx].value = parent.value;
                inList[(focusIdx-1)>>1].key = focus.key;
                inList[(focusIdx-1)>>1].value = focus.value;
                focusIdx = (focusIdx - 1)>>1;
            }
            else {
                //cuPrintf("Parent %f > focus %f\n", parent, focus);
                focusIdx = 0;
            }
        }
        if (focusIdx != 0){
            parent.key = inList[(focusIdx-1)>>1].key;
            parent.value = inList[(focusIdx-1)>>1].value;
            //Swap focus and parent if focus is bigger than parent
            if (focus.value > parent.value){
                cuPrintf("Focus %f > parent %f.  Setting index %d to focus\n",
                         focus.value, parent.value, (focusIdx-1)>>1);
                inList[focusIdx].key = parent.key;
                inList[focusIdx].value = parent.value; 
                inList[(focusIdx-1)>>1].key = focus.key;
                inList[(focusIdx-1)>>1].value = focus.value;
                focusIdx = (focusIdx-1)>>1;
            }
            else {
                //cuPrintf("Parent %f > focus %f\n", parent, focus);
                focusIdx = 0; 
            }
       }
        //localTemp = *temp;
    }
    
    //Empty the pipeline before returning
    while (focusIdx !=0){
        parent.key = inList[(focusIdx-1)>>1].key;
        parent.value = inList[(focusIdx-1)>>1].value;
        //Swap focus and parent if focus is bigger than parent
        if (focus.value > parent.value){
            cuPrintf("Focus %f > parent %f\n", focus.value, parent.value);
            inList[focusIdx].key = parent.key;
            inList[focusIdx].value = parent.value;  
            inList[(focusIdx-1)>>1].key = focus.key;
            inList[(focusIdx-1)>>1].value = focus.value;
            focusIdx = (focusIdx-1)>>1;
        }
        else {
            //cuPrintf("Parent %f > focus %f\n", parent, focus);
            focusIdx = 0; 
        }
    }
    
    return;
}

/* Pops a heap using a single warp.  Must be run on the bottom warp of a
 * thread.  If this function is not executed by all of threads 0-7, the GPU
 * will stall.
 * heap: a pointer to a heap structure w/ space for a complete heap of depth d 
 * d:  The depth of the heap 
 * count: The number of elements to pop
 */
__device__ void pipelinedPop(__volatile__ float *heap, float *out_list, 
                             int d, int popCount){
    
    int focusIdx = 0; //Index of element currently percolating down
    int maxChildIdx=0; //Index of largest child of element percolating down
    int curDepth=d+1; //Depth of element currently percolating down
    __volatile__ __shared__ int temp;
    /*int localTemp=0; Temp doesn't need to be re-read _every_ time.
                    * Temp will be used to track the next element to percolate.
                    */

    if (threadIdx.x == 0){
        temp = 0; //We have thus far popped 0 elements
    }

    //localTemp = 0;
    
    //We maintain the invariant that no two threads are processing on
    //adjacent layers of the heap in order to avoid memory conflicts and
    //race conditions.
    while (temp < popCount){
        if (threadIdx.x == (temp & 7)){
            focusIdx = 0;
            curDepth = 0;
            out_list[temp] = heap[0];
            temp = temp + 1;
            //cuPrintf("temp is: %d\n", *temp);
            //cuPrintf("top of heap is: %f\n", heap[0]);
        }
        
        //Unrolled loop once to avoid race conditions and get a small speed
        //boost over using a for loop on 2 iterations.
        if (curDepth < d-1){
            maxChildIdx = 2*focusIdx+1;
            //cuPrintf("Children are %f, %f\n", heap[2*focusIdx+2], 
            //         heap[maxChildIdx]); 
            //cuPrintf("Depth is %d, Focusing on element %d\n", curDepth,
            //         focusIdx);
            if (heap[2*focusIdx+2] > heap[maxChildIdx]){
                maxChildIdx = 2*focusIdx+2;
            }
            heap[focusIdx] = heap[maxChildIdx];
            focusIdx = maxChildIdx;
            curDepth++;
        }

        if (curDepth < d-1){
            maxChildIdx = 2*focusIdx+1;
            //cuPrintf("Depth is %d, Focusing on element %d\n", curDepth,
            //         focusIdx);
            if (heap[2*focusIdx+2] > heap[maxChildIdx]){
                maxChildIdx = 2*focusIdx+2;
            }
            heap[focusIdx] = heap[maxChildIdx];
            focusIdx = maxChildIdx;
            curDepth++;
        }

        if (curDepth == d-1){
            //cuPrintf("curDepth is %d\n", curDepth);
            //cuPrintf("focusIdx is %d\n", focusIdx);
            //cuPrintf("Depth is %d (max).  Focusing on element %d\n", curDepth,
            //focusIdx);
            heap[focusIdx] = 0;
            curDepth++;
            //continue;
        }
    }
    
    //empty the pipeline before returning
    
    while (curDepth < d-1){
        //cuPrintf("Emptying Pipeline.  Focusing on element %d\n", focusIdx); 
        maxChildIdx = 2*focusIdx+1;
        if (heap[2*focusIdx+2] > heap[maxChildIdx]){
            maxChildIdx = 2*focusIdx+2;
        }
        heap[focusIdx] = heap[maxChildIdx];
        focusIdx = maxChildIdx;
        curDepth++;
    }
    if (curDepth == d-1){
        //cuPrintf("curDepth is %d\n", curDepth);
        //cuPrintf("focusIdx is %d\n", focusIdx);
        //cuPrintf("Depth is %d (max).  Focusing on element %d\n", curDepth,
        //focusIdx);
        heap[focusIdx] = 0;
        curDepth++;
        //continue;
    }

    return;
}

void usage(){
    printf("Usage: in_list [thread_count] [kernel_count]\n"); 
}

int main(int argc, char *argv[]){
    
    int len;
    float *h_list;

    cudaPrintfInit();

    if ((argc > 4) || argc < 2) {
        printf("Invalid argument count.  %s accepts 1-4 arguments, %d given\n",
               argv[0], argc);
        usage();
        return -1;
    }
    
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    int thread_count = 64;
    //int block_count = devProp.maxGridSize[0];
    int block_count = 2;

    if (argc > 2){
        thread_count = atoi(argv[2]);
    }
    if (argc > 3){
        block_count = atoi(argv[3]);
    }

    FILE *fin = fopen(argv[1], "r");
    
    if (fin == NULL){
        printf("Could not open file: %s", argv[1]);
        return -2;
    }

    fscanf(fin, "#%d#", &len);

    h_list = (float *)malloc(len*sizeof(float));
    if (h_list == NULL){
        printf("Insufficient host memory to allocate at %d", __LINE__);
        return -3;
    }

    for (int i = 0; i < len; i++){
        if (EOF == fscanf(fin, "%f ", &h_list[i])){
            break;
        }
    }

    /*
    printf("\nInitial list is:\n");
    for (int i = 0; i < len; i++){
        printf("%f\n", h_list[i]);
    }
    */

    metaEntry_t temp[BLOCKSIZE];

    heapSort(h_list, temp, len, thread_count, block_count, devProp);

    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();

    
    printf("\nFinal list is:\n");
    for (int i = 0; i < len; i++){
        printf("%f\n", h_list[i]);
        }

    return 0;
}
