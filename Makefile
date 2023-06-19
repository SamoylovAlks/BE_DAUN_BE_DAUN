CC = g++
CFLAGS =-c -Wall

all: matrix

matrix: matrix.o 
	$(CC) -pthread matrix.o  -o matrix

matrix.o: matrix.cpp
	$(CC) $(CFLAGS) matrix.cpp


clean:
	rm -rf *.o matrix