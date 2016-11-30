CC = nvcc
EXEC = mandcu
OBJS =
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -I/usr/local/cuda/lib64
SEQFLAGS = -O3

all: $(EXEC)

mandcu: mandelbrot.cu
	$(CC) $(FLAGS) -o $@ mandelbrot.cu $(MATHFLAG)

clean: 
	rm -f *.o *.pgm $(OBJS) $(EXEC)
