CC	:= gcc
LFLAGS	:=  -lm 
CFLAGS	:= -Wall -Werror 


RM = rm -f   # rm command
TARGET_BIN = neuromz.out  # target lib


SRCS = src/neuromz.c src/main.c src/ann.c src/actFunc.c 

OBJS = $(SRCS:.c=.o)



all: ${TARGET_BIN}


$(TARGET_BIN): $(OBJS)
	$(CC) ${CFLAGS} -o $@ $^ ${LFLAGS}

$(OBJS):%.o:%.c
	$(CC) $(CFLAGS) -c $< -o $@ ${LFLAGS}




clean:
	-${RM} ${TARGET_BIN} ${OBJS}
