
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

    dataset_t paritial_set = { 
        .m = {
            [DS_RANDOM] = 0,    /* sample batch id */
            [DS_ACHP]   = 0,    /* average chlorophyll content */
            [DS_PHR]    = 0,    /* plant height growth rate */
            [DS_AWWGV]  = 0,    /* average wet weight of vegetative growth */
            [DS_ALAP]   = 0,    /* average leaf area per plant */
            [DS_ANPL]   = 0,    /* average number of leaves per plant */
            [DS_ARD]    = 0,    /* average root diameter */
            [DS_ADWR]   = 0,    /* average dry weight of roots */
            [DS_PDMVG]  = 0,    /* % dry matter in vegetative growth */
            [DS_ARL]    = 0,    /* average root length */
            [DS_AWWR]   = 0,    /* average wet weight of roots */
            [DS_ADWV]   = 0,    /* average dry weight fo vegative parts */
            [DS_PDMRG]  = 0,    /* % dry matter in root growth */
            [DS_CLASS]  = 1,    /* experiement group label */
        }
    };
    dataset_t guessed_dataset = { 0 };

    /* proccess dataset */
    read_file (dataset_csv, &dataset_contents);
    process_dataset (dataset_contents, &dataset, &dataset_count);
    free (dataset_contents);
    dataset_contents = NULL;

    /* k nearest neighbor */
    read_file (kernel_source_file, &kernel_source);
    knn_predict (kernel_source, dataset, dataset_count, paritial_set, &guessed_dataset);

    printf ("%10s %10s %10s %10s %10s %10s %10s %10s\n"
            "%10s %10s %10s %10s %10s %10s\n", 
            "RANDOM", "ACHP", "PHR",  "AWWGV", "ALAP",  "ANPL", "ARD", "ADWR", 
            "PDMVG",  "ARL",  "AWWR", "ADWV",  "PDMRG", "CLASS");

    printf ("input:\n");
    for (int i = 0; i < 16; i++)
    {
        printf ("%10.04f ", paritial_set.m[i]);
        if (((i+1) % 8) == 0)
        {
            printf ("\n");
        }
    }

    printf ("output:\n");
    for (int i = 0; i < 16; i++)
    {
        printf ("%10.4f ", guessed_dataset.m[i]);
        if (((i+1) % 8) == 0)
        {
            printf ("\n");
        }
    }

    /* clean up */
    free (kernel_source);
    kernel_source = NULL;

    free (dataset);
    dataset = NULL;
    dataset_count = 0;

    return 0;
}

/* end of file */
