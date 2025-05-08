
#include "err.h"

#include <stdio.h>

#define _(msg) (msg)

void 
error (const char *msg)
{
    (void)fprintf (stderr, "%s", _(msg));
}

/* end of file */
