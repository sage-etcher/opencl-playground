
#include "dataset.h"
#include "err.h"
#include "types.h"

#include <CL/cl.h>

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define opencl_err err

typedef struct {
    cl_platform_id platform;

    cl_device_id device;
    size_t device_core_count;
    size_t device_max_local_size;
    
    cl_context context;
    cl_program program;
    cl_kernel  kernel;
    size_t kernel_max_local_size;
    size_t kernel_preferred_local_multiple;

    cl_command_queue queue;
    cl_mem bufs[2];
} runtime_t;

#define ARRLEN(arr) (sizeof (arr) / sizeof (*(arr)))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

static void opencl_err (const char *msg);

static cl_platform_id get_default_platform (void);
static cl_device_id get_default_device (cl_platform_id platform, cl_device_type device_type);

static void init (runtime_t *state);
static void destroy (runtime_t *state);
static void create_kernel (runtime_t *state, char *kernel_name, char *source);

static void test_kernel_shader (char *source);



void 
knn_predict (const char *kernel_source, float16_t *dataset_arr, size_t dataset_count, float16_t input, float16_t *p_predicted)
{
    runtime_t state = { 0 };
    enum {
        BUF_POINTS,
        BUF_DIST,
    };

    float *dists = NULL;
    size_t iterations = dataset_count;
    int n = (uint32_t)iterations;

    dists = malloc (sizeof (float) * database_count);
    assert (dists != NULL);

    size_t global_size = 0;
    size_t local_size  = 0;

    init (&state);
    create_kernel (&state, "distance", kernel_source);

    state.queue = clCreateCommandQueue (state.context, state.device, 0, NULL);
    state.bufs[BUF_POINTS] = clCreateBuffer (state.context, CL_MEM_READ_WRITE, sizeof (float16_t) * dataset_count, NULL, NULL);
    state.bufs[BUF_DIST]   = clCreateBuffer (state.context, CL_MEM_READ_WRITE, sizeof (float) * dataset_count,  NULL, NULL);

    (void)clEnqueueWriteBuffer (state.queue, state.bufs[BUF_POINTS], CL_TRUE, 0, sizeof (float16_t) * dataset_count, dataset_arr, 0, NULL, NULL);

    (void)clSetKernelArg (state.kernel, 0, sizeof (float16_t), &pos);
    (void)clSetKernelArg (state.kernel, 1, sizeof (cl_mem),   &state.bufs[BUF_POINTS]);
    (void)clSetKernelArg (state.kernel, 2, sizeof (cl_mem),   &state.bufs[BUF_DIST]);
    (void)clSetKernelArg (state.kernel, 3, sizeof (uint32_t), &n);

    /* calculate local/global sizes */
    /* ensure local_size is the smallest possible multiple of 
     * state.kernel_preferred_local_multiple, making sure to capout at 
     * state.kernel_max_local_size. */
    size_t a = (size_t)ceil (((float)iterations / state.kernel_preferred_local_multiple));
    size_t b = a * state.kernel_preferred_local_multiple;
    local_size = MIN (b, state.kernel_max_local_size);

    /* ensure global multiple is the minimum multiple of local_size necessary 
     * to reach atleast n iterations. */
    size_t c = (size_t)ceil ((float)iterations / local_size);
    global_size = c * local_size;
    
    (void)clEnqueueNDRangeKernel (state.queue, state.kernel, 1, 0, &global_size, &local_size, 0, NULL, NULL);
    (void)clFinish (state.queue);

    (void)clEnqueueReadBuffer (state.queue, state.bufs[BUF_DIST], CL_TRUE, 0, sizeof (float) * dataset_count, dists, 0, NULL, NULL);

    for (size_t i = 0; i < iterations; i++)
    {
        printf ("%0.04f ", dists[i]);
    }

    destroy (&state);

}


static cl_platform_id
get_default_platform (void)
{
    cl_platform_id default_platform = { 0 };
    cl_platform_id *platforms = NULL;
    cl_uint platforms_count = 0;

    /* get n of platforms */
    (void)clGetPlatformIDs (0, NULL, &platforms_count);
    if (platforms_count == 0)
    {
        opencl_err ("Failed to find any platforms.");
        exit (1);
    }

    /* get platforms */
    platforms = malloc (platforms_count * sizeof (cl_platform_id));
    assert (platforms != NULL);
    (void)clGetPlatformIDs (platforms_count, platforms, NULL);

    default_platform = platforms[0];

    free (platforms);
    platforms = NULL;
    platforms_count = 0;
    return default_platform;
}


static cl_device_id
get_default_device (cl_platform_id platform, cl_device_type device_type)
{
    cl_device_id default_device = { 0 };
    cl_device_id *devices = NULL;
    cl_uint devices_count = 0;

    /* get n of devices */
    (void)clGetDeviceIDs (platform, device_type, 0, NULL, &devices_count);
    if (devices_count == 0)
    {
        opencl_err ("Failed to find any devices.");
        exit (1);
    }

    /* get platforms */
    devices = malloc (devices_count * sizeof (cl_device_id));
    assert (devices != NULL);
    (void)clGetDeviceIDs (platform, device_type, devices_count, devices, NULL);

    default_device = devices[0];

    free (devices);
    devices = NULL;
    devices_count = 0;
    return default_device;
}


static void
init (runtime_t *state)
{
    assert (state != NULL);
   
    state->platform = get_default_platform ();
    state->device   = get_default_device (state->platform, CL_DEVICE_TYPE_GPU);

    state->context = clCreateContext (NULL, 1, &state->device, NULL, NULL, NULL);
    if (state->context == NULL)
    {
        opencl_err ("Failed to create context.");
        exit (1);
    }

    (void)clGetDeviceInfo (state->device, CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof (size_t), &state->device_core_count, NULL);
    (void)clGetDeviceInfo (state->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (size_t), &state->device_max_local_size,  NULL);

    assert (state->platform != NULL);
    assert (state->device   != NULL);
    assert (state->context  != NULL);
    assert (state->device_core_count != 0);
    assert (state->device_max_local_size  != 0);
}


static void
create_kernel (runtime_t *state, char *kernel_name, char *source)
{
    cl_int err = CL_SUCCESS;

    assert (state != NULL);
    assert (source != NULL);

    state->program = clCreateProgramWithSource (state->context, 1, &source, NULL, NULL);

    err = clBuildProgram (state->program, 1, &state->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        opencl_err ("Failed to build program.");
        exit (1);
    }

    state->kernel = clCreateKernel (state->program, kernel_name, &err);
    if (err != CL_SUCCESS)
    {
        opencl_err ("Failed to create kernel.");
        exit (1);
    }

    (void)clGetKernelWorkGroupInfo (state->kernel, state->device, CL_KERNEL_WORK_GROUP_SIZE,  sizeof (size_t), &state->kernel_max_local_size,  NULL);
    (void)clGetKernelWorkGroupInfo (state->kernel, state->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof (size_t), &state->kernel_preferred_local_multiple,  NULL);

    assert (state->program != NULL);
    assert (state->kernel  != NULL);
    assert (state->kernel_max_local_size != 0);
    assert (state->kernel_preferred_local_multiple != 0);
}


static void
destroy (runtime_t *state)
{
    if (state == NULL) { return; }

    (void)clReleaseMemObject (state->bufs[0]);
    (void)clReleaseMemObject (state->bufs[1]);
    state->bufs[0] = NULL;
    state->bufs[1] = NULL;

    (void)clReleaseCommandQueue (state->queue);
    state->queue = NULL;

    (void)clReleaseKernel (state->kernel);
    state->kernel = NULL;
    state->kernel_max_local_size = 0;
    state->kernel_preferred_local_multiple = 0;

    (void)clReleaseProgram (state->program);
    state->program = NULL;

    (void)clReleaseContext (state->context);
    state->context = NULL;

    state->device = NULL;
    state->device_core_count = 0;
    state->device_max_local_size = 0;

    state->platform = NULL;
}


/* end of file */