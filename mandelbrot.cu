
/************************************\
| filename: escape.c
|
| description: sequential version
| of code that outputs a .PGM file of
| a Mandelbrot fractal.
|
| notes: the number of pixels, 2400x2400
| was chosen so that it would take a fair
| amount of time to compute the image so
| that speedup may be observed on in a parallel
| implementation.  it might be advisable
| to change the #defines for the purposes
| of developing a parallel version of the
| code.
|
| hint: the file output is a .PGM file which
| is viewable with the linux utility gimp.
| The 'convert' utility can convert
| from .pgm to .gif, which will save lots of disk
| space.
|
| authors: Bryan Schlief, Daegon Kim, Wim Bohm
|
\***********************************/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>

// CUDA includes...
#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

#define   RMIN       0.3129928802767
#define   RMAX       0.31299305009252
#define   IMIN       0.0345483210604
#define   IMAX       0.0345485012278

#define   RADIUS_SQ  4.0     /* 2^2                              */
#define   WIDTH      2400    /* # of pixels wide                 */
#define   HEIGHT     2400    /* # of pixels high                 */
#define   MAX_COLOR  UCHAR_MAX
#define   OUT_FILE   "Mandelbrot.pgm"
#define   BLOCK_SIZE 32
#define   DEF_ITER   1000
#define   DEBUG      0

void writeOutput(const char *fileName, char *data, int width, int height) {
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

#define cudaAssert(ans) { _cudaAssert((ans), __FILE__, __LINE__); }
inline void _cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess)  {
        fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void cudaPrintDevices(FILE *file, cudaDeviceProp *prop, int i) {
    fprintf(file, "Device Number: %d\n", i);
    fprintf(file, "  Device name: %s\n", prop->name);
    fprintf(file, "  Memory Clock Rate (KHz): %d\n", prop->memoryClockRate);
    fprintf(file, "  Memory Bus Width (bits): %d\n", prop->memoryBusWidth);
    fprintf(file, "  Peak Memory Bandwidth (GB/s): %f\n", 2.0* prop->memoryClockRate * (prop->memoryBusWidth / 8) / 1.0e6);
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

__global__ void mand(char* output, int maxIter, double realRange, double imagRange) {
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

        output[idx] = (char) floor(((double) (MAX_COLOR * iter)) / (double) maxIter);
    }
}

int main(int argc, char ** argv) {
    int nDevices = 0;
    cudaDeviceProp prop;

    char *hostOutput = NULL;
    char *devOutput = NULL;

    float time; 	/*timer*/

    int maxIter = DEF_ITER;
    if (argc > 1) {
        maxIter = atoi(argv[1]);    /* first command line argument... */
    }

    if (maxIter < 1) {
        printf("Usage : %s [MAX ITERATION]\n", argv[0]);
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
        for (int i = 0; i < nDevices; i++) {
            cudaAssert(cudaGetDeviceProperties(&prop, i));
            cudaPrintDevices(stderr, &prop, i);
        }
    }

    // Get data size...
    int dataSize = WIDTH * HEIGHT;
    if (DEBUG) fprintf(stderr, "dataSize = %d\n", dataSize);

    /* Allocate memory on host to store output values for pixels */
    hostOutput = (char *) calloc(dataSize, sizeof(char));
    if (hostOutput == NULL) {
        perror("hostOutput");
        return -1;
    }

    // Set block size...
    int blockWidth = BLOCK_SIZE;
    int blockHeight = blockWidth;
    dim3 blockSize(blockWidth, blockHeight);
    if (DEBUG) fprintf(stderr, "blockSize = (%d,%d)\n", blockSize.x, blockSize.y);

    // Set grid size...
    int gridWidth = WIDTH / blockSize.x;
    int gridHeight = HEIGHT / blockSize.y;
    dim3 gridSize(gridWidth, gridHeight);
    if (DEBUG) fprintf(stderr, "gridSize = (%d,%d)\n", gridSize.x, gridSize.y);

    // Create event timers...
    cudaEvent_t start, stop;
    cudaAssert(cudaEventCreate(&start));
    cudaAssert(cudaEventCreate(&stop));

    // Start timer...
    cudaEventRecord(start);

    /* Allocate memory on device... */
    if (DEBUG) fprintf(stderr, "cudaMalloc...\n");
    cudaAssert(cudaMalloc(&devOutput, dataSize * sizeof(char)));

    double realRange = (RMAX - RMIN) / (double) (WIDTH - 1);
    double imagRange = (IMAX - IMIN) / (double) (HEIGHT - 1);

    // Invoke the kernel...
    if (DEBUG) {
        fprintf(stderr, "kernel: mand(devOutput[%d], maxIter=%d, realRange=%lf, imagRange=%lf)...\n",
                dataSize, maxIter, realRange, imagRange);
    }

    mand<<<gridSize, blockSize>>>(devOutput, maxIter, realRange, imagRange);

    // Check last error...
    if (DEBUG) fprintf(stderr, "cudaPeekAtLastError...\n");
    cudaAssert(cudaPeekAtLastError());

    // Sync the device...
    //if (DEBUG) fprintf(stderr, "cudaDeviceSynchronize...\n");
    //cudaAssert(cudaDeviceSynchronize());

    // Copy data back to host
    if (DEBUG) fprintf(stderr, "cudaMemcpy...\n");
    cudaAssert(cudaMemcpy(hostOutput, devOutput, dataSize, cudaMemcpyDeviceToHost));

    // Free data on device...
    if (DEBUG) fprintf(stderr, "cudaFree...\n");
    cudaAssert(cudaFree(devOutput));

    // Stop timer...
    cudaAssert(cudaEventRecord(stop));

    // Get elapsed time...
    if (DEBUG) fprintf(stderr, "cudaEventSynchronize...\n");
    cudaAssert(cudaEventSynchronize(stop));
    if (DEBUG) fprintf(stderr, "cudaEventElapsedTime...\n");
    cudaAssert(cudaEventElapsedTime(&time, start, stop));

    // Write the output...
    if (DEBUG) fprintf(stderr, "writeOutput...\n");
    writeOutput(OUT_FILE, hostOutput, WIDTH, HEIGHT);

    // Free host data...
    free(hostOutput);

    // Report timing...
    printf("Elapsed time: %lf ms\n", time);

    return 0;
}
