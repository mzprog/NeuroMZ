CC	:= gcc
LFLAGS	:=  -lm 
CFLAGS	:= -Wall -Werror 


RM = rm -f   # rm command
PROJECT_NAME = neuromz
TARGET_BIN = neuromz.out  # target lib
BIN_LOC = /usr/local

SRCS = src/neuromz.c src/main.c src/ann.c src/actFunc.c 

OBJS = $(SRCS:.c=.o)



all: ${TARGET_BIN}


$(TARGET_BIN): $(OBJS)
	$(CC) ${CFLAGS} -o $@ $^ ${LFLAGS}

$(OBJS):%.o:%.c
	$(CC) $(CFLAGS) -c $< -o $@ ${LFLAGS}

install: ${TARGET_BIN}
	cp ${TARGET_BIN} ${PROJECT_NAME}
	install -m 0755 $(PROJECT_NAME) $(BIN_LOC)/bin
	rm $(PROJECT_NAME)



clean:
	-${RM} ${TARGET_BIN} ${OBJS}
