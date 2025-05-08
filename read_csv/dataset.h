
#ifndef DATASET_H
#define DATASET_H

#include "types.h"

#include <stddef.h>

typedef enum {
    DS_RANDOM,
    DS_ACHP,
    DS_PHR,
    DS_AWWGV,
    DS_ALAP,
    DS_ANPL,
    DS_ARD,
    DS_ADWR,
    DS_PDMVG,
    DS_ARL,
    DS_AWWR,
    DS_ADWV,
    DS_PDMRG,
    DS_CLASS,
    DS_COUNT,
} dataset_index;

typedef enum {
    DS_CLASS_NONE = 0,
    DS_CLASS_SA = 1,
    DS_CLASS_SB,
    DS_CLASS_SC,
    DS_CLASS_TA,
    DS_CLASS_TB,
    DS_CLASS_TC,
} dataset_class_t;

typedef enum {
    DS_RANDOM_NONE = 0,
    DS_RANDOM_R1 = 1,
    DS_RANDOM_R2,
    DS_RANDOM_R3,
} dataset_random_t;

typedef float16_t dataset_t;

void process_dataset (char *contents, dataset_t **p_dataset_arr, size_t *p_dataset_count);

#endif
