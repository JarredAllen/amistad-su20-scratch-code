CC         = gcc
PYTHONHLOC = /usr/include/python3.8
CFLAGS     = -I $(PYTHONHLOC) -O2

simulate.so: simulate.c
	$(CC) $(CFLAGS) -c simulate.c -o simulate.so
