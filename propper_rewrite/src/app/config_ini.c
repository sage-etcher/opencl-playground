
#include "config_ini.h"

#include "error.h"
#include "portability.h"

#include <ini.h>

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static int str_to_bool (const char *a);
static int match (const char *section, const char *name, 
                  const char *match_section, const char *match_name);
static int handler (void *user, const char *section, const char *name,
                    const char *value);

int
parse_config_file (const char *filename, config_ini_t *p_conf)
{
    config_ini_t conf = { 0 };
    assert (filename != NULL);

    if (ini_parse (filename, handler, &conf) < 0)
    {
        (void)errorf (_("can't load config from %s"), filename);
        return 1;
    }

    /* dry run if p_conf == NULL */
    if (p_conf != NULL)
    {
        *p_conf = conf;
    }

    return 0;
}


void
destroy_config_ini (config_ini_t *self)
{
    if (self == NULL) 
    { 
        return; 
    }

    free (self->dataset_file);
    free (self->dataset_format);
    free (self->input_default_input);

    (void)memset (self, 0, sizeof (config_ini_t));

    return;
}


static int
match (const char *section, const char *name, const char *match_section, 
       const char *match_name)
{
    return ((strcmp (section, match_section) == 0) &&
            (strcmp (name, match_name) == 0));
}

static int
str_to_bool (const char *a)
{
    if ((strcmp (a, "true") == 0) ||
        (strcmp (a, "True") == 0) ||
        (strcmp (a, "TRUE") == 0))
    {
        return 1;
    }
    else if ((strcmp (a, "false") == 0) ||
             (strcmp (a, "False") == 0) ||
             (strcmp (a, "FALSE") == 0))
    {
        return 0;
    }

    return -1;
}

static int 
handler (void *user, const char *section, const char *name, const char *value)
{
    config_ini_t *p_conf = (config_ini_t *)user;

    if (match (section, name, "dataset", "file"))
    {
        p_conf->dataset_file = strdup (value);
        assert (p_conf->dataset_file != NULL);
    }
    else if (match (section, name, "dataset", "has_headers"))
    {
        p_conf->dataset_has_headers = str_to_bool (value);
    }
    else if (match (section, name, "dataset", "format"))
    {
        p_conf->dataset_format = strdup (value);
        assert (p_conf->dataset_format != NULL);
    }
    else if (match (section, name, "input", "default_input"))
    {
        p_conf->input_default_input = strdup (value);
        assert (p_conf->input_default_input != NULL);
    }
    else if (match (section, name, "knn", "k"))
    {
        p_conf->knn_k = atoi (value);
    }
    else
    {
        /* error state */
        return 0;
    }

    return 1;
}


/* end of file */