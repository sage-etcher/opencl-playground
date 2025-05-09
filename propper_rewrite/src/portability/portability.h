
#ifndef PORTABILITY_H
#define PORTABILITY_H

/* MSVC prefers _strdup */
#ifdef _MSC_VER
#   define strdup _strdup
#endif

#endif