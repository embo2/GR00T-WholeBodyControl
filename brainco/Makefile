CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -ldl

all: hand_server

hand_server: hand_server.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lzmq -lmsgpackc

clean:
	rm -f hand_server

.PHONY: all clean
