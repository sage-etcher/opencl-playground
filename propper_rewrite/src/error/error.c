
#include "error.h"
 
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int
vfprefixf (FILE *fp, const char *prefix, const char *fmt, va_list args)
{
    const size_t buf_n = 1024;
    char buf[1024];

    (void)vsnprintf (buf, buf_n, fmt, args);

    return fprintf (fp, "%s: %s\n", prefix, buf);
}

int
fprefixf (FILE *fp, const char *prefix, const char *fmt, ...)
{
    int rc = 0;
    va_list args = NULL;
    va_start (args, &fmt);

    rc = vfprefixf (fp, prefix, fmt, args);

    va_end (args);
    args = NULL;

    return rc;
}

int
vferrorf (FILE *fp, const char *fmt, va_list args)
{
    return vfprefixf (fp, "error", fmt, args);
}

int
ferrorf (FILE *fp, const char *fmt, ...)
{
    int rc = 0;
    va_list args = NULL;
    va_start (args, &fmt);

    rc = vferrorf (fp, fmt, args);

    va_end (args);
    args = NULL;

    return rc;
}

int
verrorf (const char *fmt, va_list args)
{
    return vferrorf (stderr, fmt, args);
}

int
errorf (const char *fmt, ...)
{
    int rc = 0;
    va_list args = NULL;
    va_start (args, &fmt);

    rc = verrorf (fmt, args);

    va_end (args);
    args = NULL;

    return rc;
}

void
fatal (void)
{
    errorf (_("a fatal error has occured, aborting..."));
    exit (EXIT_FAILURE);
}

/* end of file */