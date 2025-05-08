
PROG=   knntest
SRCS=   dataset.c err.c knn.c main.c read_file.c
MAN=    

CFLAGS=     -Wall -Wextra
CFLAGS+=    -Wno-unused-variable
CFLAGS+=    -Wno-unused-parameter
CFLAGS+=    -DCL_TARGET_OPENCL_VERSION=300
#CFLAGS+=    -DNDEBUG
#CFLAGS+=    -DDEBUG_DEFAULT_SHADER
COPT=   -O0 -ggdb

LDFLAGS=    -lOpenCL -lm

.include <bsd.prog.mk>
