
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
            //[DS_RANDOM] =  1.00,    /* sample batch id */
            [DS_ACHP]   = 34.21,    /* average chlorophyll content */
            [DS_PHR]    = 64.99,    /* plant height growth rate */
            [DS_AWWGV]  =  1.24,    /* average wet weight of vegetative growth */
            [DS_ALAP]   = 1476,     /* average leaf area per plant */
            [DS_ANPL]   =  3.90,    /* average number of leaves per plant */
            [DS_ARD]    = 16.30,    /* average root diameter */
            [DS_ADWR]   = 0.794,    /* average dry weight of roots */
            [DS_PDMVG]  = 43.69,    /* % dry matter in vegetative growth */
            [DS_ARL]    = 20.35,    /* average root length */
            [DS_AWWR]   =  2.80,    /* average wet weight of roots */
            [DS_ADWV]   =  0.55,    /* average dry weight fo vegative parts */
            [DS_PDMRG]  = 27.81,    /* % dry matter in root growth */
            //[DS_CLASS]  =  2.00,    /* experiement group label */
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

    printf ("%10s  %10s    %10s\t%s\n", "Short Name", "Input", "Output", "Description");
    printf ("-------------------------------------------------------------\n");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "RANDOM", paritial_set.m[0],  guessed_dataset.m[0],  "Sample Batch ID");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "ACHP",   paritial_set.m[1],  guessed_dataset.m[1],  "Average Chlorophyll Content");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "PHR",    paritial_set.m[2],  guessed_dataset.m[2],  "Plant Height Growth Rate");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "AWWGV",  paritial_set.m[3],  guessed_dataset.m[3],  "Average Wet-Weight of Vegetative Growth");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "ALAP",   paritial_set.m[4],  guessed_dataset.m[4],  "Average Leaf Area per Plant");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "ANPL",   paritial_set.m[5],  guessed_dataset.m[5],  "Average Number of Leaves per Plant");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "ARD",    paritial_set.m[6],  guessed_dataset.m[6],  "Average Root Diameter");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "ADWR",   paritial_set.m[7],  guessed_dataset.m[7],  "Average Dry-Weight of Roots");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "PDMVG",  paritial_set.m[8],  guessed_dataset.m[8],  "Percent Dry Matter in Vegetative Growth");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "ARL",    paritial_set.m[9],  guessed_dataset.m[9],  "Average Root Length");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "AWWR",   paritial_set.m[10], guessed_dataset.m[10], "Average Wet-Weight of Roots");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "ADWV",   paritial_set.m[11], guessed_dataset.m[11], "Average Dry-Weight of Roots");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "PDMRG",  paritial_set.m[12], guessed_dataset.m[12], "Percent Dry Matter in Root Growth");
    printf ("%10s: %10.04f -> %10.04f\t%s\n", "CLASS",  paritial_set.m[13], guessed_dataset.m[13], "Experiement Group ID");

    /* clean up */
    free (kernel_source);
    kernel_source = NULL;

    free (dataset);
    dataset = NULL;
    dataset_count = 0;

    return 0;
}

/* end of file */
