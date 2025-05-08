
#include "err.h"

#include <stdio.h>

#define _(msg) (msg)


void 
err (const char *msg)
{
    (void)fprintf (stderr, "%s", _(msg));
}
