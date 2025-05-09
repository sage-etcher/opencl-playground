
#include "config.h"
#include "config_ini.h"
#include "error.h"

#include <stdio.h>
#include <stdlib.h>

int
main (int argc, char **argv)
{
    config_ini_t conf = { 0 };
    
    if (parse_config_file (GLOBAL_CONFIG, &conf))
    {
        fatal ();
    }

    printf ("%s %s %s %d %d\n", 
            conf.dataset_file,
            conf.dataset_format,
            conf.input_default_input,
            conf.dataset_has_headers,
            conf.knn_k);
    
    destroy_config_ini (&conf);
    return 0;
}
/* end of file */