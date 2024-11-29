#CC= /opt/local/bin/gcc-mp-13

all: csolve.so cgauss_seidel.so

csolve.so: csolve.o
	gcc -shared -o $@ csolve.o -lm

cgauss_seidel.so: cgauss_seidel.o csolve.o
	gcc -shared -o $@ $^ -lm

#CFLAGS= -fPIC -Ofast -ftree-loop-vectorize -fopt-info-optimized  -fopt-info-vec-missed  -march=znver2
#-fopt-info-optimized -fopenmp
CFLAGS= -fPIC -Ofast -ftree-loop-vectorize -ffast-math -fopt-info-optimized -funroll-all-loops 
CFLAGS= -fPIC -O3 -ftree-loop-vectorize -funroll-all-loops  -freciprocal-math -ffast-math -fopt-info-optimized
#CFLAGS= -fPIC -O0 -g

.SUFFIXES : .o .c 
.c.o    :
	$(CC) -c $(CFLAGS) $(OPTC) $<

clean:
	rm -f *.o

deepclean:
	rm -rf *.o *.so *.h5 __pycache__

