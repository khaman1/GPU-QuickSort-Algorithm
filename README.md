QuickSort-Algorithm
===================
--------------------------------------------------------------------------------
SYSTEM REQUIREMENTS
--------------------------------------------------------------------------------

1. OS: linux, Windows

2. CPU: all

3. GPU: NVIDIA Fermi GPU architecture or higher (1.x capability or higher)

4  builder: CUDA 4.0 or higher

   note: It is recommended to use NVIDIA Kepler GPU architecture with CUDA 5.0; 
	 using CUDA 4.0 or 4.2 would result in a slower runtime

--------------------------------------------------------------------------------
HOW TO BUILD : LINUX
--------------------------------------------------------------------------------

1. The makefile can be found in the Release_linux folder

2. Type 'make'.  This builds the Fast Parallel Quick-Sort:

	make all   - build all projects (the executable file will be created in the Relese_linux folder)
	make clean - clean project
	
--------------------------------------------------------------------------------
HOW TO RUN ON LINUX
--------------------------------------------------------------------------------
The binaries are exported in the folder "bin"
You can test with CPU_Quicksort by the following command:
	./CPU_QuickSort 1000 numbersInt1000
Or with GPU-QuickSort
	./GPU_quicksort numbersInt1000
Moreover, you can plot cache misses, hits, ... by the following command:
	bpsh 12 nvprof  --events  l1_global_load_hit,l1_global_load_miss ./bin/GPU_quicksort ./bin/numbersInt1000
