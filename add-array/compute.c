
#include "compute.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <Cl/cl.h>

static size_t get_max_work_group_size (cl_device_id device);

compute_t *
compute_init (void)
{
    cl_int err = CL_SUCCESS;
    compute_t *self = NULL;
    
    self = calloc (1, sizeof (compute_t));
    assert (self != NULL);

    err |= compute_default_platform (self);
    err |= compute_default_device (self, self->default_platform);
    err |= compute_create_context (self, self->num_devices, self->all_devices);

    if (err != CL_SUCCESS)
    {
        (void)fprintf (stderr, "OpenCL: Failed to initialize a new compute_t object\n");
        compute_destroy (self);
        self = NULL;
        return self;
    }

    self->max_global_size = get_max_work_group_size (self->default_device);

    return self;
}

void 
compute_destroy (compute_t *self)
{
    if (self == NULL) { return; }

    (void)clReleaseKernel (self->kernel);
    (void)clReleaseProgram (self->program);

    (void)clReleaseContext (self->context);

    free (self->all_devices);
    self->all_devices = NULL;
    self->num_devices = 0;

    free (self->all_platforms);
    self->all_platforms = NULL;
    self->num_platforms = 0;

    free (self);
}

cl_int 
compute_find_platforms (compute_t *self)
{
    cl_int err = CL_SUCCESS;
    cl_platform_id *all_platforms = NULL;
    cl_uint         num_platforms = 0;

    assert (self != NULL);

    err |= clGetPlatformIDs (0, NULL, &num_platforms);
    if (err != CL_SUCCESS)
    {
        return err;
    }

    if (num_platforms == 0)
    {
        return err;
    }
    all_platforms = malloc (num_platforms * sizeof (cl_platform_id));
    assert (all_platforms != NULL);

    err |= clGetPlatformIDs (num_platforms, all_platforms, NULL);
    if (err != CL_SUCCESS)
    {
        free (all_platforms);
        all_platforms = NULL;
        num_platforms = 0;
        return err;
    }

    self->all_platforms = all_platforms;
    self->num_platforms = num_platforms;
   
    return err;
}

cl_int 
compute_find_devices (compute_t *self, cl_platform_id platform)
{
    cl_int err = CL_SUCCESS;
    cl_device_id *all_devices = NULL;
    cl_uint       num_devices = 0;

    assert (self != NULL);

    err |= clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS)
    {
        return err;
    }

    if (num_devices == 0)
    {
        return err;
    }
    all_devices = malloc (num_devices * sizeof (cl_device_id));
    assert (all_devices != NULL);

    err |= clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, num_devices, all_devices, NULL);
    if (err != CL_SUCCESS)
    {
        free (all_devices);
        all_devices = NULL;
        num_devices = 0;
        return err;
    }

    self->all_devices = all_devices;
    self->num_devices = num_devices;
   
    return err;

}

cl_int 
compute_default_platform (compute_t *self)
{
    cl_int err = CL_SUCCESS;

    assert (self != NULL);

    err |= compute_find_platforms (self);
    if (self->num_platforms == 0)
    {
        (void)fprintf (stderr, "OpenCL: Cannot find any platform.\n");
        return err;
    }

    self->default_platform = self->all_platforms[0];

    return err;
}

cl_int 
compute_default_device (compute_t *self, cl_platform_id platform)
{
    cl_int err = CL_SUCCESS;

    assert (self != NULL);

    err |= compute_find_devices (self, platform);
    if (self->num_devices == 0)
    {
        (void)fprintf (stderr, "OpenCL: Cannot find any devices.\n");
        return err;
    }

    self->default_device = self->all_devices[0];

    return err;
}

cl_int
compute_create_context (compute_t *self, cl_uint num_devices, cl_device_id *all_devices)
{
    cl_int err = CL_SUCCESS;
    cl_context context = NULL;

    assert (self != NULL);
    assert (all_devices != NULL);
    assert (num_devices != 0);

    context = clCreateContext (NULL, num_devices, all_devices, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        (void)fprintf (stderr, "OpenCL: Cannot create context.\n");
        return err;
    }

    self->context = context;
    
    return err;
}

cl_int 
compute_create_command_queue (compute_t *self)
{
    cl_int err = CL_SUCCESS;
    cl_command_queue queue = NULL;

    assert (self != NULL);
    assert (self->context != NULL);
    assert (self->default_device!= NULL);

    queue = clCreateCommandQueue (self->context, self->default_device, 0, &err);
    if (err != CL_SUCCESS)
    {
        return err;
    }

    self->queue = queue;

    return err;
}

cl_int 
compute_allocate_n_bufs (compute_t *self, size_t n)
{
    void *tmp = NULL;

    assert (self != NULL);
    assert (n != 0);

    tmp = realloc (self->bufs, n * sizeof (cl_mem));
    if (tmp == NULL)
    {
        return -1;
    }

    self->bufs     = tmp;
    self->num_bufs = n;

    return CL_SUCCESS;
}

cl_int 
compute_create_buf (compute_t *self, cl_uint id, cl_uint buf_size)
{
    cl_int err = CL_SUCCESS;
    cl_mem buf = NULL;

    assert (self != NULL);
    assert (buf_size != 0);
    assert (self->context != NULL);
    assert (self->bufs != NULL);
    assert (self->num_bufs >= id);

    buf = clCreateBuffer (self->context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
    if (err != CL_SUCCESS)
    {
        return err;
    }

    self->bufs[id] = buf;

    return err;
}

cl_int 
compute_write_buf (compute_t *self, cl_uint id, cl_uint buf_size, void *buf)
{
    cl_int err = CL_SUCCESS;

    assert (self != NULL);
    assert (self->num_bufs > id);
    assert (self->bufs != NULL);
    assert (self->queue != NULL);
    assert (buf != NULL);
    assert (buf_size != 0);

    err = clEnqueueWriteBuffer (self->queue, self->bufs[id], CL_TRUE, 0, buf_size, buf, 0, NULL, NULL);

    return err;
}

cl_int 
compute_read_buf (compute_t *self, cl_uint id, cl_uint buf_size, void *buf)
{
    cl_int err = CL_SUCCESS;

    assert (self != NULL);
    assert (self->num_bufs > id);
    assert (self->bufs != NULL);
    assert (self->queue != NULL);
    assert (buf != NULL);
    assert (buf_size != 0);

    err = clEnqueueReadBuffer (self->queue, self->bufs[id], CL_TRUE, 0, buf_size, buf, 0, NULL, NULL);

    return err;
}

cl_int 
compute_create_kernel (compute_t *self, const char *kernel_name, size_t lines, const char **kernel_code)
{
    cl_int err = CL_SUCCESS;
    cl_program program = NULL;
    cl_kernel  kernel  = NULL;

    assert (self != NULL);
    assert (kernel_code != NULL);
    assert (self->context != NULL);

    program = clCreateProgramWithSource (self->context, lines, kernel_code, NULL, &err);
    err |= clBuildProgram (program, 1, &self->default_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        (void)fprintf (stderr, "OpenCL: Error building program\n");
        (void)clReleaseProgram (program);
        return err;
    }

    kernel = clCreateKernel (program, kernel_name, &err);
    if (err != CL_SUCCESS)
    {
        (void)fprintf (stderr, "OpenCL: Error creating Kernel\n");
        (void)clReleaseProgram (program);
        return err;
    }

    self->program = program;
    self->kernel  = kernel;

    return err;
}

cl_int
compute_execute (compute_t *self, size_t iterations)
{
    cl_int err = CL_SUCCESS;
    cl_uint i = 0;

    size_t global_size = 0;
    size_t local_size = 0;

    assert (self != NULL);
    assert (self->kernel != NULL);
    assert (self->queue != NULL);
    assert (self->bufs != NULL);
    assert (self->num_bufs != 0);
    assert (self->max_global_size != 0);
    
    global_size = (size_t)fmin ((float)iterations, (float)self->max_global_size);
    local_size  = (size_t)ceil ((float)iterations / global_size);

    printf ("%zu %zu %zu %zu\n", iterations, self->max_global_size, global_size, local_size);

    for (; i < self->num_bufs; i++)
    {
        err |= clSetKernelArg (self->kernel, i, sizeof (cl_mem), &self->bufs[i]);
        if (err != CL_SUCCESS)
        {
            return err;
        }
    }

    err |= clEnqueueNDRangeKernel (self->queue, self->kernel, 1, 0, &global_size, &local_size, 0, NULL, NULL);
    err |= clFinish (self->queue);

    return err;
}

static size_t
get_max_work_group_size (cl_device_id device)
{
    size_t max_work_group_size = 0;
    
    (void)clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (size_t), &max_work_group_size, NULL);

    return max_work_group_size;
}


/* end of file */