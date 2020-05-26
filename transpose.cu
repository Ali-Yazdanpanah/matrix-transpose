#include "matrix.h"
#include <cuda.h>
// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of
// BLOCK_ROWS

#define TILE_DIM 16
#define BLOCK_ROWS 16
#define BLOCK_DIM 16

#define NUM_REPS 10



double_matrix *newDoubleMatrix(int rows, int cols) {
  if (rows <= 0 || cols <= 0)
    return NULL;

  // allocate a matrix structure
  double_matrix *m = (double_matrix *)malloc(sizeof(double_matrix));

  // set dimensions
  m->rows = rows;
  m->cols = cols;

  // allocate a double array of length rows * cols
  m->data = (double *)malloc(rows * cols * sizeof(double));
  // set all data to 0
  int i;
  srand(time(NULL)); // Initialization, should only be called once.
  for (i = 0; i < rows * cols; i++)
    m->data[i] = (double)rand();

  return m;
}

int_matrix *newIntMatrix(int rows, int cols) {
  if (rows <= 0 || cols <= 0)
    return NULL;

  // allocate a matrix structure
  int_matrix *m = (int_matrix *)malloc(sizeof(int_matrix));

  // set dimensions
  m->rows = rows;
  m->cols = cols;

  // allocate a double array of length rows * cols
  m->data = (int *)malloc(rows * cols * sizeof(int));
  // set all data to 0
  int i;
  srand(time(NULL)); // Initialization, should only be called once.
  for (i = 0; i < rows * cols; i++)
    m->data[i] = rand();

  return m;
}

// pointer to element in matrix by row and column location
#define ELEM(mtx, row, col) mtx->data[(col - 1) * mtx->rows + (row - 1)]

/* Prints the matrix to stdout.  Returns 0 if successful
 * and -1 if mtx is NULL.
 */
int printDoubleMatrix(double_matrix *mtx) {
  if (!mtx)
    return -1;
  int row, col;
  for (row = 1; row <= mtx->rows; row++) {
    for (col = 1; col <= mtx->cols; col++) {
      // Print the floating-point element with
      //  - either a - if negative or a space if positive
      //  - at least 3 spaces before the .
      //  - precision to the hundredths place
      printf("% 6.2f ", ELEM(mtx, row, col));
    }
    // separate rows by newlines
    printf("\n");
  }
  return 0;
}

int printIntMatrix(int_matrix *mtx) {
  if (!mtx)
    return -1;
  int row, col;
  for (row = 1; row <= mtx->rows; row++) {
    for (col = 1; col <= mtx->cols; col++) {
      // Print the floating-point element with
      //  - either a - if negative or a space if positive
      //  - at least 3 spaces before the .
      //  - precision to the hundredths place
      printf("%d ", ELEM(mtx, row, col));
    }
    // separate rows by newlines
    printf("\n");
  }
  return 0;
}

/* Writes the transpose of matrix in into matrix out.
 * Returns 0 if successful, -1 if either in or out is NULL,
 * and -2 if the dimensions of in and out are incompatible.
 */
int transposeDouble(double_matrix *in, double_matrix *out) {
  if (!in || !out)
    return -1;
  if (in->rows != out->cols || in->cols != out->rows)
    return -2;

  int row, col;
  for (row = 1; row <= in->rows; row++)
    for (col = 1; col <= in->cols; col++)
      ELEM(out, col, row) = ELEM(in, row, col);
  return 0;
}

int transposeInt(int_matrix *in, int_matrix *out) {
  if (!in || !out)
    return -1;
  if (in->rows != out->cols || in->cols != out->rows)
    return -2;

  int row, col;
  for (row = 1; row <= in->rows; row++)
    for (col = 1; col <= in->cols; col++)
      ELEM(out, col, row) = ELEM(in, row, col);
  return 0;
}

/* Sets the (row, col) element of mtx to val.  Returns 0 if
 * successful, -1 if mtx is NULL, and -2 if row or col are
 * outside of the dimensions of mtx.
 */
int setDoubleElement(double_matrix *mtx, int row, int col, double val) {
  if (!mtx)
    return -1;
  assert(mtx->data);
  if (row <= 0 || row > mtx->rows || col <= 0 || col > mtx->cols)
    return -2;

  ELEM(mtx, row, col) = val;
  return 0;
}

int setIntElement(int_matrix *mtx, int row, int col, int val) {
  if (!mtx)
    return -1;
  assert(mtx->data);
  if (row <= 0 || row > mtx->rows || col <= 0 || col > mtx->cols)
    return -2;

  ELEM(mtx, row, col) = val;
  return 0;
}

/* Copies a matrix.  Returns NULL if mtx is NULL.
 */

double_matrix *copyDoubleMatrix(double_matrix *mtx) {
  if (!mtx)
    return NULL;

  // create a new matrix to hold the copy
  double_matrix *cp = newDoubleMatrix(mtx->rows, mtx->cols);

  // copy mtx's data to cp's data
  memcpy(cp->data, mtx->data, mtx->rows * mtx->cols * sizeof(double));

  return cp;
}

int_matrix *copyIntMatrix(int_matrix *mtx) {
  if (!mtx)
    return NULL;

  // create a new matrix to hold the copy
  int_matrix *cp = newIntMatrix(mtx->rows, mtx->cols);

  // copy mtx's data to cp's data
  memcpy(cp->data, mtx->data, mtx->rows * mtx->cols * sizeof(int));

  return cp;
}

/* Deletes a matrix.  Returns 0 if successful and -1 if mtx
 * is NULL.
 */
int deleteDoubleMatrix(double_matrix *mtx) {
  if (!mtx)
    return -1;
  // free mtx's data
  assert(mtx->data);
  free(mtx->data);
  // free mtx itself
  free(mtx);
  return 0;
}

int deleteIntMatrix(int_matrix *mtx) {
  if (!mtx)
    return -1;
  // free mtx's data
  assert(mtx->data);
  free(mtx->data);
  // free mtx itself
  free(mtx);
  return 0;
}

__global__ void transposeDiagonalInt(int *odata, int *idata, int width,
                                     int height) {
  __shared__ int tile[TILE_DIM][TILE_DIM + 1];

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  if (width == height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x * blockIdx.y;
    blockIdx_y = bid % gridDim.y;
    blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
  }

  // from here on the code is same as previous kernel except blockIdx_x replaces
  // blockIdx.x and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
  }

  __syncthreads();

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }
}

__global__ void transposeDiagonalDouble(double *odata, double *idata, int width,
                                        int height) {
  __shared__ double tile[TILE_DIM][TILE_DIM + 1];

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  if (width == height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x * blockIdx.y;
    blockIdx_y = bid % gridDim.y;
    blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
  }

  // from here on the code is same as previous kernel except blockIdx_x replaces
  // blockIdx.x and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
  }

  __syncthreads();

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }
}

int main(int argc, char **argv) {

  struct timeval start, end;
  if (argc != 3) {
    printf("Correct way to execute this program is:\n");
    printf("./transpose-serial NumberOfRows NumberOfColumns.\n");
    return 1;
  }
  int rows = atoi(argv[1]);
  int columns = atoi(argv[2]);
  int data_size = rows * columns;
  int *d_m2_arr, *d_m2Trans_arr;
  double *d_m1_arr, *d_m1Trans_arr;

  // creating two matrices, 1 with double values and the other with single
  // values
  double_matrix *m1 = newDoubleMatrix(rows, columns);
  int_matrix *m2 = newIntMatrix(rows, columns);

  // creating transpose matrices
  double_matrix *m1Trans = newDoubleMatrix(rows, columns);
  int_matrix *m2Trans = newIntMatrix(rows, columns);
  int_matrix *r2 = newIntMatrix(rows, columns);

  // setup execution parameters
  dim3 grid(columns / BLOCK_DIM, rows / BLOCK_DIM, 1);
  dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

  // initialize data on device
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_m2_arr, sizeof(int) * data_size));
  CUDA_CHECK_RETURN(
      cudaMalloc((void **)&d_m2Trans_arr, sizeof(int) * data_size));
  CUDA_CHECK_RETURN(cudaMemcpy(d_m2_arr, m2->data, sizeof(int) * data_size,
                               cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_m1_arr, sizeof(double) * data_size));
  CUDA_CHECK_RETURN(
      cudaMalloc((void **)&d_m1Trans_arr, sizeof(double) * data_size));
  CUDA_CHECK_RETURN(cudaMemcpy(d_m1_arr, m1->data, sizeof(double) * data_size,
                               cudaMemcpyHostToDevice));

  // transpose kernels
  gettimeofday(&start, NULL);
  for (int i = 0; i < NUM_REPS; i++) {
    transposeDiagonalInt<<<grid, threads>>>(d_m2Trans_arr, d_m2_arr, columns,
                                            rows);
  }
  CUDA_CHECK_RETURN(
      cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
  gettimeofday(&end, NULL);
  double diffInt =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
  printf("Transpose of matrix with int values took: %.4fms\n",
         diffInt / (1000 * NUM_REPS));

  gettimeofday(&start, NULL);
  for (int j = 0; j < NUM_REPS; j++) {
    transposeDiagonalDouble<<<grid, threads>>>(d_m1Trans_arr, d_m1_arr, columns,
                                               rows);
  }
  CUDA_CHECK_RETURN(
      cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
  gettimeofday(&end, NULL);
  double diffDouble =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
  printf("Transpose of matrix with double values took: %.4fms\n",
         diffDouble / (1000 * NUM_REPS));

  CUDA_CHECK_RETURN(cudaGetLastError());
  CUDA_CHECK_RETURN(cudaMemcpy(m2Trans->data, d_m2Trans_arr,
                               sizeof(int) * data_size,
                               cudaMemcpyDeviceToHost));

  CUDA_CHECK_RETURN(cudaMemcpy(m1Trans->data, d_m1Trans_arr,
                               sizeof(double) * data_size,
                               cudaMemcpyDeviceToHost));

  // free cuda
  cudaFree(d_m2Trans_arr);
  cudaFree(d_m2_arr);
  cudaFree(d_m1Trans_arr);
  cudaFree(d_m1_arr);
  // deleteing matrices

  cudaDeviceReset();

  deleteDoubleMatrix(m1);
  deleteIntMatrix(m2);
  deleteDoubleMatrix(m1Trans);
  deleteIntMatrix(m2Trans);
}
