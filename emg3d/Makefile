#CC= /opt/local/bin/gcc-mp-13

all: ccore.so 

ccore.so: ccore.o
	gcc -shared -o $@ ccore.o -lm

#CFLAGS= -fPIC -Ofast -ftree-loop-vectorize -fopt-info-optimized  -fopt-info-vec-missed  -march=znver2
#-fopt-info-optimized -fopenmp
CFLAGS= -fPIC -Ofast -ftree-loop-vectorize -ffast-math -fopt-info-optimized -funroll-all-loops 
CFLAGS= -fPIC -O3 -ftree-loop-vectorize -funroll-all-loops 
#CFLAGS= -fPIC -O0 -g

.SUFFIXES : .o .c 
.c.o    :
	$(CC) -c $(CFLAGS) $(OPTC) $<

clean:
	rm -f ccore *.o

deepclean:
	rm -rf ccore *.o *.so *.h5 __pycache__

