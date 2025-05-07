
#include "compute.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#define ARRLEN(arr) (sizeof(arr) / sizeof(*(arr)))

static void print_iarr (int *arr, size_t n);
static void log_device_info (cl_device_id device);

int main (void)
{
    cl_int err = 0;
    compute_t *opencl = NULL;
    int arr_a[30] = {  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int arr_b[30] = { 10, 9, 7, 1, 2, 5, 9, 3, 0, 3, 10, 9, 7, 1, 2, 5, 9, 3, 0, 3, 10, 9, 7, 1, 2, 5, 9, 3, 0, 3 };
    int arr_c[30] = { 0 };
    
    int iterations = ARRLEN(arr_a);

    enum {
        BUF_A,
        BUF_B,
        BUF_C,
        BUF_N,
        BUF_COUNT,
    };
    
    const char *kernel_code = 
        "void kernel simple_add (global const int *a, global const int *b, global int *c, global const int *n)"
        "{"
        "    int id = get_global_id (0);"
        "    if (id >= *n)"
        "    {"
        "        return;"
        "    }"
        "    c[id] = a[id] + b[id];"
        "}";

    opencl = compute_init ();
    log_device_info (opencl->default_device);

    (void)compute_create_kernel (opencl, "simple_add", 1, &kernel_code);

    (void)compute_create_command_queue (opencl);

    (void)compute_allocate_n_bufs (opencl, BUF_COUNT);
    (void)compute_create_buf (opencl, BUF_A, sizeof (arr_a));
    (void)compute_create_buf (opencl, BUF_B, sizeof (arr_b));
    (void)compute_create_buf (opencl, BUF_C, sizeof (arr_c));
    (void)compute_create_buf (opencl, BUF_N, sizeof (iterations));

    (void)compute_write_buf (opencl, BUF_A, sizeof (arr_a), arr_a);
    (void)compute_write_buf (opencl, BUF_B, sizeof (arr_b), arr_b);
    (void)compute_write_buf (opencl, BUF_N, sizeof (iterations), &iterations);

    (void)compute_execute (opencl, (size_t)iterations);

    (void)compute_read_buf (opencl, BUF_C, sizeof (arr_c), arr_c);

    printf ("  ");
    print_iarr (arr_a, ARRLEN(arr_a));
    printf ("\n");
    printf ("+ ");
    print_iarr (arr_b, ARRLEN(arr_b));
    printf ("\n");
    printf ("--------------------------------------------------------------------------------------------\n");
    printf ("  ");
    print_iarr (arr_c, ARRLEN(arr_c));
    printf ("\n");

    compute_destroy (opencl);
    return 0;
}


static void
print_iarr (int *arr, size_t n)
{
    for (; n > 0; n--, arr++)
    {
        printf ("%2d ", *arr);
    }
}

static void 
log_device_info (cl_device_id device)
{
    cl_uint max_compute_units = 0;
    cl_uint max_work_item_dimensions = 0;
    size_t max_work_group_size = 0;
    
    (void)clGetDeviceInfo (device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof (cl_uint), &max_compute_units, NULL);
    (void)clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof (cl_uint), &max_work_item_dimensions, NULL);
    (void)clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (size_t), &max_work_group_size, NULL);

    (void)fprintf (stdout, "MAX_COMPUTE_UNTIS:        %u\n", (unsigned)max_compute_units);
    (void)fprintf (stdout, "MAX_WORK_ITEM_DIMENSIONS: %u\n", (unsigned)max_work_item_dimensions);
    (void)fprintf (stdout, "MAX_WORK_GROUP_SIZE:      %u\n", (unsigned)max_work_group_size);
}

/* end of file */
