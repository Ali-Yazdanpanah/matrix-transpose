all: transpose-serial transpose-parallel

transpose-serial: matrix.h transpose.c
	gcc -pg -std=c99 transpose.c -o transpose-serial

transpose-parallel: matrix.h transpose.cu
	nvcc -o transpose-parallel transpose.cu 

clean:
	rm transpose-serial transpose-parallel