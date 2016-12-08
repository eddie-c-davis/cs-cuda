#!/usr/bin/env python3

# Author:  Eddie C. Davis
# Program: Mandelbrot
# Project: MPI
# File:    mand-cuda-wrap.py

import os
import sys
import subprocess as sub
import traceback as tb
import filecmp

DEBUG = False
MIN_VALID_TIME = 0.01
THREADS_PER_BLOCK = 1024

def run(args, stream=None):
    if DEBUG:
        print(' '.join(args))

    data = sub.check_output(args, env=os.environ, stderr=sub.STDOUT)
    output = data.decode()

    if stream is not None:
        stream.write(output)

    return output


def main():
    exec = '/home/edavis/Work/Courses/530/cuda/trunk/mandcu'
    nIter = 10000
    nRuns = 5
    inc = 32
    xMin = 0
    yMin = 0
    xMax = 1024
    yMax = xMax
    imgWidth= 2400
    imgHeight = imgWidth

    args = sys.argv[1:]
    nArgs = len(args)
    if nArgs > 0:
        exec = args[0]
    if nArgs > 1:
        nIter = int(args[1])
    if nArgs > 2:
        inc = int(args[2])
    if nArgs > 3:
        nRuns = int(args[3])
    if nArgs > 4:
        xMin = int(args[4])
    if nArgs > 5:
        xMax = int(args[5])
    if nArgs > 6:
        yMin = int(args[6])
    if nArgs > 7:
        yMax = int(args[7])

    if '-h' in exec:
        print("usage: mand-cuda-wrap.py <executable> <max_iterations> <img_width> <img_height> <increment> <runs_per_case> <min_block_x> <max_block_x> <min_block_y> <max_block_y>")
        return

    path = os.path.dirname(exec)
    outFile = '%s/Mandelbrot.pgm' % path
    refFile = '%s/Mandelbrot-ref.pgm' % path

    print('BlockX,BlockY,GridX,GridY,MaxIter,Time')

    if DEBUG:
        print("xMin = %d, yMin = %d, xMax = %d, yMax = %d" %  (xMin, yMin, xMax, yMax))

    blockX = xMin
    while blockX < xMax + 1:
        blockY = yMin
        nThreads = blockX * blockY
        if DEBUG:
            print("blockX = %d, blockY = %d, nThreads = %d" % (blockX, blockY, nThreads))
        
        while blockY < yMax + 1:
            if blockX > 0:
                gridX = imgWidth // blockX
            else:
                gridX = imgWidth

            if blockY > 0:
                gridY = imgHeight // blockY
            else:
                gridY = imgHeight

            if blockX == 0 and blockY == 0:
                nThreads = THREADS_PER_BLOCK + 1    # Error state...

            if nThreads <= THREADS_PER_BLOCK:
                # Invoke CUDA C program
                args = [exec, '%d' % nIter, '%d' % imgWidth, '%d' % imgHeight, '%d' % blockX, '%d' % blockY]

                isValid = True

                mtime = 0.0
                for i in range(nRuns):
                    output = run(args)

                    # Diff output file with reference file on 1st iteration, only report times for runs with valid output...
                    if i == 0:
                        isValid = filecmp.cmp(outFile, refFile, False)
                        if not isValid:
                            break

                    lines = output.rstrip().split("\n")
                    mtime += float(lines[-1].split()[-2])

                if isValid:
                    mtime /= float(nRuns)
                    if mtime  >= MIN_VALID_TIME:
                        print('%d,%d,%d,%d,%d,%lf' % (blockX, blockY, gridX, gridY, nIter, mtime))
                        sys.stdout.flush()

            blockY += inc

        blockX += inc


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt as e: # Ctrl-C
        print("Closing gracefully on keyboard interrupt...")

    except Exception as e:
        print('Unexpected Exception: ' + str(e))
        tb.print_exc()
        os._exit(1)

