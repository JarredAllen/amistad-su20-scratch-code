CC         = gcc
PYTHONHLOC = /usr/include/python3.8
CFLAGS     = -I $(PYTHONHLOC) -Og

simulate.so: simulate.c
	$(CC) $(CFLAGS) -shared -fPIC simulate.c -o simulate.so
