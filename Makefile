CC = nvcc
EXEC = mandcu
OBJS =
MATHFLAG = -lm
FLAGS = -O3 -I/usr/local/cuda/include -Wno-deprecated-gpu-targets
#FLAGS = -g -G -I/usr/local/cuda/include -Wno-deprecated-gpu-targets

all: $(EXEC)

mandcu: mandelbrot.cu
	$(CC) $(FLAGS) -o $@ mandelbrot.cu $(MATHFLAG)

clean: 
	rm -f *.o *.pgm $(OBJS) $(EXEC)
