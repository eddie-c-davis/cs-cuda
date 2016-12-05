#!/usr/bin/env python3

# Author:  Eddie C. Davis
# Program: Mandelbrot
# Project: MPI
# File:    mand-cuda-wrap.py

import os
import sys
import subprocess
import traceback as tb

def run(args, stream=None):
    data = subprocess.check_output(args, env=os.environ, stderr=subprocess.STDOUT)
    output = data.decode()

    if stream is not None:
        stream.write(output)

    return output


def main():
    exec = '/home/edavis/Work/Courses/530/cuda/trunk/mandcu'
    nIter = 10000
    nRuns = 5
    inc = 32
    xMin = 32
    yMin = 0
    xMax = 512
    yMax = xMax
    imgSize = 2400

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
        print("usage: mand-cuda-wrap.py <executable> <max_iterations> <increment> <runs_per_case> <min_block_x> <max_block_x> <min_block_y> <max_block_y>")
        return

    print('BlockX,BlockY,GridX,GridY,MaxIter,Time')

    blockX = xMin
    while blockX <= xMax:
        blockY = yMin
        while blockY <= yMax:
            gridX = imgSize // blockX
            if blockY > 0:
                gridY = imgSize // blockY
            else:
                gridY = imgSize

            # Invoke CUDA C program
            args = [exec, '%d' % nIter, '%d' % blockX, '%d' % blockY]

            mtime = 0.0
            for i in range(nRuns):
                output = run(args)
                lines = output.rstrip().split("\n")
                mtime += float(lines[-1].split()[-2])

            mtime /= float(nRuns)
            print('%d,%d,%d,%d,%d,%lf' % (blockX, blockY, gridX, gridY, nIter, mtime))

            blockY += inc
            if blockY > yMax:
                blockY = yMax

        blockX += inc
        if blockX > xMax:
            blockX = xMax



if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt as e: # Ctrl-C
        print("Closing gracefully on keyboard interrupt...")

    except Exception as e:
        print('Unexpected Exception: ' + str(e))
        tb.print_exc()
        os._exit(1)
