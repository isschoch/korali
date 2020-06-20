all: jacobi.o
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`

jacobi.o: jacobi.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
