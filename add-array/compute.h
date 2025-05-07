
#ifndef COMPUTE_H
#define COMPUTE_H

#include <Cl/cl.h>

typedef struct
{
    cl_platform_id *all_platforms;
    cl_uint         num_platforms;

    cl_device_id *all_devices;
    cl_uint       num_devices;

    cl_platform_id default_platform;
    cl_device_id   default_device;

    size_t max_global_size;

    cl_context context;
    cl_program program;
    cl_kernel  kernel;
    cl_command_queue queue;

    cl_mem *bufs;
    cl_uint num_bufs;
} compute_t;

compute_t *compute_init (void); 
void compute_destroy (compute_t *self);

cl_int compute_find_platforms (compute_t *self);
cl_int compute_find_devices (compute_t *self, cl_platform_id platform);

cl_int compute_default_platform (compute_t *self);
cl_int compute_default_device (compute_t *self, cl_platform_id platform);

cl_int compute_create_context (compute_t *self, cl_uint num_devices, cl_device_id *all_devices);

cl_int compute_create_command_queue (compute_t *self);
cl_int compute_allocate_n_bufs (compute_t *self, size_t n);
cl_int compute_create_buf (compute_t *self, cl_uint id, cl_uint buf_size);
cl_int compute_write_buf (compute_t *self, cl_uint id, cl_uint buf_size, void *buf);
cl_int compute_read_buf (compute_t *self, cl_uint id, cl_uint buf_size, void *buf);

cl_int compute_create_kernel (compute_t *self, const char *kernel_name, size_t lines, const char **kernel_code);
cl_int compute_execute (compute_t *self, size_t iterations);


#endif
/* end of file */