
#include "dataset.h"
#include "read_file.h"
#include "knn.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


/* main */
int
main (int argc, char **argv)
{
    const char *kernel_source_file = "euclidean_distance.cl";
    char *kernel_source = NULL;

    const char *dataset_csv = "Greenhouse Plant Growth Metrics.csv";
    char *dataset_contents = NULL;
    dataset_t *dataset = NULL;
    size_t dataset_count = 0;

    dataset_t guessed_dataset = { 0 };
    dataset_t paritial_set = { 0 };
    paritial_set.m[DS_ACHP]  = 33.0;
    paritial_set.m[DS_PHR]   = 60.0;
    paritial_set.m[DS_CLASS] = DS_CLASS_TC;

    /* proccess dataset */
    read_file (dataset_csv, &dataset_contents);
    process_dataset (dataset_contents, &dataset, &dataset_count);
    free (dataset_contents);
    dataset_contents = NULL;

    /* k nearest neighbor */
    read_file (kernel_source_file, &kernel_source);
    knn_predict (kernel_source, dataset, dataset_count, paritial_set, &guessed_dataset);

    /* clean up */
    free (kernel_source);
    kernel_source = NULL;

    free (dataset);
    dataset = NULL;
    dataset_count = 0;

    return 0;
}

/* end of file */