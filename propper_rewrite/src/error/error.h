 
#ifndef ERROR_H
#define ERROR_H

#include <stdarg.h>
#include <stdio.h>

#define _(m) (m)

int vfprefixf (FILE *fp, const char *prefix, const char *fmt, va_list args);
int  fprefixf (FILE *fp, const char *prefix, const char *fmt, ...);

int vferrorf (FILE *fp, const char *fmt, va_list args);
int  ferrorf (FILE *fp, const char *fmt, ...);

int verrorf (const char *fmt, va_list args); 
int  errorf (const char *fmt, ...);

void fatal (void);

#endif