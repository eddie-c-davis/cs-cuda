/**
 * @author Eddie Davis (eddiedavis@u.boisestate.edu)
 * @author Jeff Pope (jeffreymithoug@u.boisestate.edu)
 * @file mandelbrot.cu
 * @brief CS530 PA4: Mandelbrot-CUDA Impementation
 * @date 12/4/2016
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <cuda_runtime.h>

#define   RMIN       0.3129928802767
#define   RMAX       0.31299305009252
#define   IMIN       0.0345483210604
#define   IMAX       0.0345485012278

#define   RADIUS_SQ  4.0                /* 2^2  */
#define   WIDTH      2400               /* Image width in pixels */
#define   HEIGHT     2400               /* Image height in pixels */
#define   MAX_COLOR  UCHAR_MAX          /* 255 */
#define   BLOCK_SIZE 32                 /* BLOCK_SIZE = GCD(WIDTH, THREADS_PER_BLOCK) = GCD(2400, 1024) */
#define   MIN_BLK_SZ 1
#define   OUT_FILE   "Mandelbrot.pgm"
#define   DEF_ITER   1000
#define   DEBUG      0

typedef unsigned char BYTE;

/**
 * writeOutput
 *
 * Writes Mandelbrot image in PGM format.
 *
 * @param fileName Filename to write PGM data.
 * @param data Output array data (Mandelbrot pixels)
 * @param width Image width
 * @param height Image height
 */
void writeOutput(const char *fileName, BYTE *data, int width, int height) {
    int i, j;      /* index variables */
    int max = -1;  /* for pgm file output */
    int size = width * height;

    /* PGM file format requires the largest pixel value, calculate this */
    for (i = 0; i < size; ++i) {
        if (data[i] > max) {
            max = data[i];
        }
    }

    /* open the file for writing. omit error checking. */
    FILE * fout = fopen(fileName, "w");

    /* PGM file header */
    fprintf(fout, "P2\n");
    fprintf(fout, "%d\t%d\n", width, height);
    fprintf(fout, "%d\n",max);

    /* throw out the data */
    for (i = 0; i < height; ++i) {
        for (j = 0; j < width; ++j) {
            fprintf(fout, "%d\t", data[i * width + j]);
        }

        fprintf(fout,"\n");
    }

    /* flush the buffer and close the file */
    fflush(fout);
    fclose(fout);
}

/**
 * cudaAssert
 *
 * CUDA error handler.
 *
 * @param code cudaError_t error code struct.
 * @param file Name of file in which error occurred.
 * @param line Line number on which error occurred.
 */
#define cudaAssert(ans) { _cudaAssert((ans), __FILE__, __LINE__); }
inline void _cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess)  {
        fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/**
 * cudaPrintDevice
 *
 * Prints revelevant information about the given CUDA device.
 *
 * @param file File pointer to write device properties.
 * @param prop cudaDeviceProp structure pointer.
 * @param dnum CUDA device number.
 */
void cudaPrintDevice(FILE *file, cudaDeviceProp *prop, int dnum) {
    fprintf(file, "Device Number: %d\n", dnum);
    fprintf(file, "  Device name: %s\n", prop->name);
    fprintf(file, "  Memory Clock Rate (KHz): %d\n", prop->memoryClockRate);
    fprintf(file, "  Memory Bus Width (bits): %d\n", prop->memoryBusWidth);
    fprintf(file, "  Peak Memory Bandwidth (GB/s): %f\n",
            2.0 * prop->memoryClockRate * (prop->memoryBusWidth / 8) / 1.0e6);
    fprintf(file, "  Compute Version: %d.%d\n", prop->major, prop->minor);
    fprintf(file, "  Compute Mode: ");

    switch (prop->computeMode) {
        case cudaComputeModeExclusive:
            fprintf(file, "Exclusive");
            break;
        case cudaComputeModeProhibited:
            fprintf(file, "Prohibited");
            break;
        default:
            fprintf(file, "Default");
            break;
    }

    fprintf(file, "\n");
    fprintf(file, "  SM count: %d\n", prop->multiProcessorCount);
    fprintf(file, "  Shared mem/block: %zd\n", prop->sharedMemPerBlock);
    fprintf(file, "  Threads per warp: %d\n", prop->warpSize);
    fprintf(file, "  Max threads per block: %d\n", prop->maxThreadsPerBlock);

    fprintf(file, "  Max block size: (");
    for (int j = 0; j < 3; j++) {
        fprintf(file, "%d,", prop->maxThreadsDim[j]);
    }

    fprintf(file, ")\n  Max grid size: (");
    for (int j = 0; j < 3; j++) {
        fprintf(file, "%d,", prop->maxGridSize[j]);
    }

    fprintf(file, ")\n\n");

}

/**
 * mand (CUDA kernel function)
 *
 * Generates the Mandelbrot set.
 *
 * @param output Output array to receive computed Mandelbrot pixels.
 * @param maxIter Max iterations to test for escape values.
 * @param realRange Range of real component.
 * @param imagRange Range of imaginary component.
 */
__global__ void mand(BYTE* output, int maxIter, double realRange, double imagRange) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;  // Image col (X coord)
    int row = blockDim.y * blockIdx.y + threadIdx.y;  // Image row (Y coord)

    if (col < WIDTH && row < HEIGHT) {
        int idx = row * WIDTH + col;

        double cReal = RMIN + row * realRange;
        double cImag = IMIN + col * imagRange;

        double zReal = 0.0;
        double zImag = 0.0;
        double zReal2 = zReal;
        double zImag2 = zImag;
        double zCurr;
        double zMag;

        int iter = 0;
        for (; iter < maxIter; ++iter) {
            zCurr = zReal;

            zReal2 = zReal * zReal;
            zImag2 = zImag * zImag;

            zReal = zReal2 - zImag2 + cReal;
            zImag = (2.0 * zCurr * zImag) + cImag;

            zMag = zReal2 + zImag2;
            if (zMag > RADIUS_SQ) {
                break;
            }
        }

        output[idx] = (BYTE) floor(((double) (MAX_COLOR * iter)) / (double) maxIter);
    }
}

/**
 * main
 *
 * Main function.
 *
 * @param argc Argument count.
 * @param argv Argument values.
 * @return
 */
int main(int argc, char ** argv) {
    int nDevices = 0;

    BYTE *output = NULL;
    BYTE *d_output = NULL;

    float time; 	/*timer*/

    int maxIter = DEF_ITER;
    if (argc > 1) {
        maxIter = atoi(argv[1]);    /* first command line argument... */
    }

    if (maxIter < 1) {
        printf("usage: %s [MAX_ITERATION=%d] [BLOCK_X=%d] [BLOCK_Y=1]\n", argv[0], DEF_ITER, BLOCK_SIZE);
        return 0;
    }

    printf("Running Mandelbrot-CUDA with %d iterations...\n", maxIter);

    cudaAssert(cudaGetDeviceCount(&nDevices));
    if (nDevices < 1) {
        printf("ERROR: No valid CUDA devices on this machine!\n");
        return -1;
    }

    if (DEBUG) {
        fprintf(stderr, "nDevices = %d\n", nDevices);
        cudaDeviceProp prop;
        for (int i = 0; i < nDevices; i++) {
            cudaAssert(cudaGetDeviceProperties(&prop, i));
            cudaPrintDevice(stderr, &prop, i);
        }
    }

    // Get data size...
    int dataSize = WIDTH * HEIGHT;
    if (DEBUG) fprintf(stderr, "dataSize = %d\n", dataSize);

    /* Allocate memory on host to store output values for pixels */
    output = (BYTE *) calloc(dataSize, sizeof(BYTE));
    if (output == NULL) {
        perror("output");
        return -1;
    }

    // Set block size...
    int blockX = 0;
    if (argc > 2) {
        blockX = atoi(argv[2]);
    }

    if (blockX < 1) {
        blockX = BLOCK_SIZE;
    }

    int blockY = 0;
    if (argc > 3) {
        blockY = atoi(argv[3]);
    }

    if (blockY < 1) {
        blockY = MIN_BLK_SZ;
    }

    dim3 blockSize(blockX, blockY);
    if (DEBUG) fprintf(stderr, "blockSize = (%d,%d,%d)\n", blockSize.x, blockSize.y, blockSize.z);

    // Set grid size...
    int gridX = WIDTH / blockSize.x;
    int gridY = HEIGHT / blockSize.y;
    dim3 gridSize(gridX, gridY);
    if (DEBUG) fprintf(stderr, "gridSize = (%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z);

    // Create event timers...
    cudaEvent_t start, stop;
    cudaAssert(cudaEventCreate(&start));
    cudaAssert(cudaEventCreate(&stop));

    // Start timer...
    cudaAssert(cudaEventRecord(start));

    // Allocate memory on device...
    if (DEBUG) fprintf(stderr, "cudaMalloc...\n");
    cudaAssert(cudaMalloc(&d_output, dataSize * sizeof(char)));

    double realRange = (RMAX - RMIN) / (double) (WIDTH - 1);
    double imagRange = (IMAX - IMIN) / (double) (HEIGHT - 1);

    // Invoke the kernel...
    if (DEBUG) {
        fprintf(stderr, "kernel: mand(d_output[%d], maxIter=%d, realRange=%lf, imagRange=%lf)...\n",
                dataSize, maxIter, realRange, imagRange);
    }

    mand<<<gridSize, blockSize>>>(d_output, maxIter, realRange, imagRange);

    // cudaMemcpy is an implicit barrier so need need for sync.

    // Copy data back to host...
    if (DEBUG) fprintf(stderr, "cudaMemcpy...\n");
    cudaAssert(cudaMemcpy(output, d_output, dataSize, cudaMemcpyDeviceToHost));

    // Free data on device...
    if (DEBUG) fprintf(stderr, "cudaFree...\n");
    cudaAssert(cudaFree(d_output));

    // Stop timer...
    cudaAssert(cudaEventRecord(stop));

    // Get elapsed time...
    if (DEBUG) fprintf(stderr, "cudaEventSynchronize...\n");
    cudaAssert(cudaEventSynchronize(stop));
    if (DEBUG) fprintf(stderr, "cudaEventElapsedTime...\n");
    cudaAssert(cudaEventElapsedTime(&time, start, stop));

    // Write the output...
    if (DEBUG) fprintf(stderr, "writeOutput...\n");
    writeOutput(OUT_FILE, output, WIDTH, HEIGHT);

    // Free host data...
    free(output);

    // Report timing...
    printf("Elapsed time: %lf sec\n", time * 1E-3);

    return 0;
}
