CC = g++
FLAGS = -std=c++11
CVFLAGS = `pkg-config --cflags --libs opencv`

all: video.cpp
	$(CC) $(FLAGS) video.cpp -o video $(CVFLAGS)

