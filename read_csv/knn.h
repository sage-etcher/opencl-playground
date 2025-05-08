
#ifndef KNN_H
#define KNN_H

#include "types.h"

#include <stddef.h>

void knn_predict (const char *kernel_source, float16_t *dataset_arr, size_t dataset_count, float16_t input, float16_t *p_predicted);

#endif