
#include "dataset.h"

#include "err.h"

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void 
process_dataset (char *contents, dataset_t **p_dataset_arr, size_t *p_dataset_count)
{
    size_t i = 0;
    char *data = NULL;
    char *iter = NULL;
    dataset_t *ds = NULL;
    size_t ds_entry = 0;
    size_t ds_elem = 0;
    float ds_data = 0.0;

    assert (contents != NULL);
    assert (p_dataset_arr != NULL);
    assert (p_dataset_count != NULL);

    /* skip first line header */
    data = strchr (contents, '\n');
    data++;

    /* count # of newlines */
    iter = data;
    i = 0;
    while ((iter != NULL) && (*iter != '\0'))
    {
        if (*iter == '\n')
        {
            i++;
        }
        iter ++;
    }

    /* allocate room for all entrys */
    ds = calloc (i, sizeof (dataset_t));
    assert (ds != NULL);

        /* proccess the csv file */
    iter = strtok (data, "\n,");
    while (iter != NULL)
    {
        /* parse the element into number */
        switch (ds_elem)
        {
        case DS_RANDOM:
            if (strcmp (iter, "R1") == 0)
            {
                ds_data = DS_RANDOM_R1;
            }
            else if (strcmp (iter, "R2") == 0)
            {
                ds_data = DS_RANDOM_R2;
            }
            else if (strcmp (iter, "R3") == 0)
            {
                ds_data = DS_RANDOM_R3;
            }
            else
            {
                error ("DS_RANDOM out of range.");
                exit (1);
            }
            break;

        case DS_CLASS:
            if (strcmp (iter, "SA") == 0)
            {
                ds_data = DS_CLASS_SA;
            }
            else if (strcmp (iter, "SB") == 0)
            {
                ds_data = DS_CLASS_SB;
            }
            else if (strcmp (iter, "SC") == 0)
            {
                ds_data = DS_CLASS_SC;
            }
            else if (strcmp (iter, "TA") == 0)
            {
                ds_data = DS_CLASS_TA;
            }
            else if (strcmp (iter, "TB") == 0)
            {
                ds_data = DS_CLASS_TB;
            }
            else if (strcmp (iter, "TC") == 0)
            {
                ds_data = DS_CLASS_TC;
            }
            else
            {
                error ("DS_CLASS out of range.");
                exit (1);
            }
            break;

        case DS_ACHP:
        case DS_PHR:
        case DS_AWWGV:
        case DS_ALAP:
        case DS_ANPL:
        case DS_ARD:
        case DS_ADWR:
        case DS_PDMVG:
        case DS_ARL:
        case DS_AWWR:
        case DS_ADWV:
        case DS_PDMRG:
            ds_data = atof (iter);
            break;

        default:
            error ("parser read out of range.");
            exit (1);
        }

        /* add the element to dataset */
        ds[ds_entry].m[ds_elem] = ds_data;

        /* loop */
        ds_elem++;
        if (ds_elem >= DS_COUNT)
        {
            ds_elem = 0;
            ds_entry++;
        }
        iter = strtok (NULL, "\n,");
    }

    *p_dataset_arr = ds;
    *p_dataset_count = ds_entry; 
    return;
}
