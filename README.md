# PA4 Part 1 CUDA Mandelbrot README

Eddie Davis (eddiedavis@u.boisestate.edu)

Jeff Pope (jeffreymithoug@u.boisestate.edu)

This is a CUDA GPU implementation of the Mandelbrot set calculation and image generation.

NOTE: Must be built on a system with CUDA installed.

# 1) Contents

    mandelbrot.cu    CUDA C source file.
    Makefile         Makefile that invokes nvcc to compile the program.
    README.md        This README file.

# 2) Building

make clean; make
rm -f *.o *.pgm  mandcu
nvcc -O3 -I/usr/local/cuda/include -Wno-deprecated-gpu-targets --compiler-bindir /home/faculty/cathie/gcc-4.7/bin -o mandcu mandelbrot.cu -lm

# 3) Running

    a. Run mandcu executale with -h flag to get help message.
    $ ./mandcu -h

    usage: mandcu [MAX_ITERATION=1000] [BLOCK_X=32] [BLOCK_Y=BLOCK_X]

    b. Run with specified command line arguments (or none for defaults).

    $ ./mandcu

    Running Mandelbrot-CUDA with 1000 iterations...
    Elapsed time: 1.511489 sec

    c. Check the output image for correctness.

    $ gimp Mandelbrot.pgm


Thank you!
