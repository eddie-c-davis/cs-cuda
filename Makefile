CC = nvcc
EXEC = mandcu
OBJS =
MATHFLAG = -lm
FLAGS = -O3 -I/usr/local/cuda/lib64 -Wno-deprecated-gpu-targets
#FLAGS = -g -G -keep -I/usr/local/cuda/lib64 -Wno-deprecated-gpu-targets

all: $(EXEC)

mandcu: mandelbrot.cu
	$(CC) $(FLAGS) -o $@ mandelbrot.cu $(MATHFLAG)

clean: 
	rm -f *.o *.pgm $(OBJS) $(EXEC)
