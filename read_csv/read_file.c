
#include "read_file.h"

#include "err.h"

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>


void 
read_file (const char *file, char **p_content)
{
    long pos = 0;
    size_t size = 0;
    size_t read_count = 0;
    char *content = NULL;
    FILE *fp = NULL;

    assert (file != NULL);
    assert (p_content != NULL);

    /* open the file */
    fp = fopen (file, "r");
    if (fp == NULL)
    {
        err ("Failed to open file.");
        exit (1);
    }

    /* get the file size */
    (void)fseek (fp, 0L, SEEK_END);
    pos = ftell (fp);
    if (pos < 0)
    {
        err ("Error getting length of file.");
        (void)fclose (fp);
        fp = NULL;
        exit (1);
    }
    size = (size_t)pos;

    /* read the file */
    content = malloc (size + 1);
    assert (content != NULL);
    (void)fseek (fp, 0L, SEEK_SET);
    read_count = fread (content, sizeof (char), size, fp);
    content[read_count] = '\0';

    /* close the file and exit */
    (void)fclose (fp);
    fp = NULL;

    *p_content = content;
    return;
}
