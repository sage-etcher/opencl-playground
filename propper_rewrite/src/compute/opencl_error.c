
#include "error.h"

#include <stdarg.h>
#include <stdio.h>

int 
opencl_err (const char *fmt, ...)
{
    int rc = 0;
    va_list args = NULL;
    va_start (args, &fmt);

    rc = vfprefixf (stderr, "OpenCL error", fmt, args);

    va_end (args);
    args = NULL;

    return rc;
}