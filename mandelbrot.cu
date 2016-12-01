
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
#include <sys/time.h>

// CUDA includes...
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define   RADIUS_SQ  4.0     /* 2^2                              */
#define   WIDTH    2400    /* # of pixels wide                 */
#define   HEIGHT    2400    /* # of pixels high                 */
#define   MAX_COLOR  255

double timer() {
    struct timeval time;
    double tval = 0.0;

//#ifdef _OPENMP
//    tval = omp_get_wtime();
//#else
    gettimeofday(&time, NULL);
    tval = ((double) time.tv_sec) + ((double) time.tv_usec) / 1E6;
//#endif

    return tval;
}


void writeOutput(char *fileName, int * data, int width, int height) {
    int i, j;    /* index variables */
    int max=-1;  /* for pgm file output */

    /* PGM file format requires the largest */
    /* pixel value.  Calculate this.        */
    for(i=0; i<width*height; ++i) {
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
    for(i=0; i<height; ++i) {
        for(j=0; j<width; ++j) {
            fprintf(fout, "%d\t", data[i*width+j]);
        }
        fprintf(fout,"\n");
    }

    /* flush the buffer and close the file */
    fflush(fout);
    fclose(fout);
}

#define cudaAssert(ans) { _cudaAssert((ans), __FILE__, __LINE__); }
inline void _cudaAssert(cudaError_t code, char *file, int line) {
    if (code != cudaSuccess)  {
        fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void cudaPrintDeviceProperties(FILE *file, cudaDeviceProp *prop, int i) {
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

__global__ void mand(int* buffer, int maxIter) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Image col (X coord)
    int row = blockDim.y * blockIdx.y + threadIdx.y;  // Image row (Y coord)

    if (col < WIDTH && row < HEIGHT) {
        int idx = row * WIDTH + col;

        // Do some math...
        float x0 = ((float) col / WIDTH) * 3.5f - 2.5f;
        float y0 = ((float) row / HEIGHT) * 3.5f - 1.75f;

        float x = 0.0f;
        float y = 0.0f;
        int iter = 0;

        float xtemp;
        for (int iter = 0; (x * x + y * y <= 4.0f) && (iter < maxIter); iter++) {
            xtemp = x * x - y * y + x0;
            y = 2.0f * x * y + y0;
            x = xtemp;
        }

        int color = iter * 5;
        if (color > MAX_COLOR) {
            color = 0;
        }

        buffer[idx] = color;
    }
}

int main(int argc, char ** argv) {
    int i,j;                            /* index variables */
    int counter;                       /* measures the "speed" at which a particular point diverges.    */
    int nDevices = 0;
    int tid = 0, tmax = 1;

    float time; 	/*timer*/

    double real_max, real_min, imag_max, imag_min;          /* varibles that define the 'c' plane; */

    double real_range, imag_range;      /* distance per pixel */

    double c_real, c_imag,              /* c and z variables  */
          z_real, z_imag, z_magnitude; /* for inner for loop */

    double z_current_real;              /* temporary variable that holds the value */
                                       /* of z_real for the calculation of z_imag  */
    cudaDeviceProp prop;

    if ( argc < 2 ) {
        printf("Usage : %s [MAX ITERATION]\n", argv[0]);
        exit(0);
    }

    int maxIter = atoi(argv[1]);    /* first command line argument... */

    char* outFileName = "Mandelbrot.pgm";  /* the sequential output filename */

    int *hostOutput = NULL;
    int *devOutput = NULL;

    cudaAssert(cudaGetDeviceCount(&nDevices));
    printf("nDevices = %d\n", nDevices);
    for (i = 0; i < nDevices; i++) {
        cudaAssert(cudaGetDeviceProperties(&prop, i));
        cudaPrintDeviceProperties(stdout, &prop, i);
    }

    int dataSize = WIDTH * HEIGHT;
    fprintf(stderr, "dataSize = %d\n", dataSize);

    // Set block size...
    int blockWidth = 32;
    int blockHeight = blockWidth;
    dim3 blockSize(blockWidth, blockHeight);
    fprintf(stderr, "blockSize = (%d,%d)\n", blockSize.x, blockSize.y);

    // Set grid size...
    int gridWidth = WIDTH / blockSize.x;
    int gridHeight = HEIGHT / blockSize.y;
    dim3 gridSize(gridWidth, gridHeight);
    fprintf(stderr, "gridSize = (%d,%d)\n", gridSize.x, gridSize.y);

    // Create event timers...
    cudaEvent_t start, stop;
    cudaAssert(cudaEventCreate(&start));
    cudaAssert(cudaEventCreate(&stop));

    /* Allocate memory on host to store output values for pixels */
    hostOutput = (int *) calloc(dataSize, sizeof(int));
    if (hostOutput == NULL) {
        perror("hostOutput");
        return -1;
    }

    // Start timer...
    cudaEventRecord(start);

    /* Allocate memory on device... */
    fprintf(stderr, "cudaMalloc...\n");
    cudaAssert(cudaMalloc(&devOutput, dataSize * sizeof(int)));

    // Invoke the kernel...
    fprintf(stderr, "kernel mand...\n");
    mand<<<gridSize, blockSize>>>(devOutput, maxIter);

    // Check last error...
    fprintf(stderr, "cudaPeekAtLastError...\n");
    cudaAssert(cudaPeekAtLastError());

    // Sync the device...
    fprintf(stderr, "cudaDeviceSynchronize...\n");
    cudaAssert(cudaDeviceSynchronize());

    // Copy data back to host
    fprintf(stderr, "cudaMemcpy...\n");
    cudaAssert(cudaMemcpy(hostOutput, devOutput, dataSize, cudaMemcpyDeviceToHost));

    // Free data on device...
    fprintf(stderr, "cudaFree...\n");
    cudaAssert(cudaFree(devOutput));

    // Stop timer...
    cudaAssert(cudaEventRecord(stop));

    // Get elapsed time...
    fprintf(stderr, "cudaEventSynchronize...\n");
    cudaAssert(cudaEventSynchronize(stop));
    fprintf(stderr, "cudaEventElapsedTime...\n");
    cudaAssert(cudaEventElapsedTime(&time, start, stop));

    // Write the output...
    fprintf(stderr, "writeOutput...\n");
    writeOutput(outFileName, hostOutput, WIDTH, HEIGHT);

    // Free host data...
    free(hostOutput);

    // Report timing...
    printf("Elapsed time: %lf ms\n", time);

//    real_min = 0.3129928802767, real_max =  0.31299305009252;  /* define the 'c' plane */
//    imag_min = 0.0345483210604, imag_max =  0.0345485012278;   /* you can change these for fun */
//
//    real_range = (real_max - real_min) / (WIDTH - 1);
//    imag_range = (imag_max - imag_min) / (HEIGHT - 1);
//
//    time = timer();
//
//    fprintf(stderr, "tmax=%d\n", tmax);
//    timers = (double *) calloc(tmax, sizeof(double));
//
//   for (i = 0; i < HEIGHT; ++i) {
//       timers[tid] = timer() - time;
//
//      for (j = 0; j < WIDTH; ++j) {
//        c_real = real_min + i * real_range;
//        c_imag = imag_min + j * imag_range;
//
//        z_real = 0.0;
//        z_imag = 0.0;
//
//        for(counter = 0; counter < MAX_ITER; ++counter) {
//           z_current_real = z_real;
//
//           z_real = (z_real * z_real) - (z_imag * z_imag) + c_real;
//           z_imag = (2.0 * z_current_real * z_imag) + c_imag;
//
//           z_magnitude = (z_real * z_real) + (z_imag * z_imag);
//
//           if(z_magnitude > RADIUS_SQ) {
//              break;
//           }
//        }  //end for
//
//        output[i*WIDTH+j] = (int)floor(((double)(MAX_COLOR * counter)) / (double)MAX_ITER);
//      } // end for
//   } // end for

    return 0;
}
