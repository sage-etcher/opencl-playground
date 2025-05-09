
#ifndef CONFIG_INI_H
#define CONFIG_INI_H

#include <stddef.h>

typedef struct {
    char *dataset_file;
    char *dataset_format;
    int   dataset_has_headers;

    char *input_default_input;

    int knn_k;
} config_ini_t;

int parse_config_file (const char *filename, config_ini_t *conf);
void destroy_config_ini (config_ini_t *self);

#endif 