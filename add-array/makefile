
PROG=   cltest
SRCS=   main.c compute.c
MAN=    

CFLAGS=     -Wall -Wextra
CFLAGS+=    -Wno-unused-variable
CFLAGS+=    -Wno-unused-parameter
CFLAGS+=    -DCL_TARGET_OPENCL_VERSION=300
COPT=   -O0 -ggdb

LDFLAGS=-lOpenCL

.include <bsd.prog.mk>
