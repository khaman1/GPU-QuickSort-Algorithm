all:
	nvcc common/main.cu common/parser.cu common/debug.cu radixsort.cu GPU_quicksort.cu CPU_GPU_quicksort.cu -c -arch sm_20
	nvcc CPU_GPU_quicksort.cu -arch sm_20 -o bin/CPU_GPU_quicksort
	nvcc GPU_quicksort.cu -arch sm_20 -o bin/GPU_quicksort
	nvcc heapsort.cu  -arch sm_20 -o bin/heapsort
	g++ quick_sort.c -o bin/CPU_QuickSort
	rm -f *.o
	rm -f *~
	rm -f *.out

clean:
	rm -f *~
	rm -f *.o
	rm -f main
	rm -f common/*~
	rm -f common/*.o
