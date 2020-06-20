all: heat2d.o
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`

heat2d.o: heat2d.cpp 
	$(KORALICXX) -c $< $(KORALICFLAGS) 

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
