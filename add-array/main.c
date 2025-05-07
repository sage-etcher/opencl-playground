
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#define ARRLEN(arr) (sizeof(arr) / sizeof(*(arr)))

typedef struct {
    cl_platform_id *all_platforms;
    cl_uint         num_platforms;
    cl_platform_id default_platform;

    cl_device_id *all_devices;
    cl_uint       num_devices;
    cl_device_id default_device;

    cl_context context;
} cl_setup_t;

static void log_platform_name (cl_platform_id platform, char *msg);
static void log_device_name (cl_device_id device, char *msg);
static cl_int get_platforms (cl_platform_id **p_all_platforms, cl_uint *p_num_platforms);
static cl_int get_devices (cl_platform_id platform, cl_device_id **p_all_devices, cl_uint *p_num_devices);

static int initialize_cl (cl_setup_t *self);
static void destroy_cl (cl_setup_t *self);

static void print_iarr (int *arr, size_t n);

int main (void)
{
    cl_int err = 0;
    cl_setup_t opencl = { 0 };
    int arr_a[10] = {  0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int arr_b[10] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    int arr_c[10] = { 0 };

    size_t size = ARRLEN(arr_a);

    size_t global_size = size;
    size_t local_size  = 1;



    (void)initialize_cl (&opencl);

    cl_mem buf_a = clCreateBuffer (opencl.context, CL_MEM_READ_WRITE, sizeof (arr_a), NULL, NULL);
    cl_mem buf_b = clCreateBuffer (opencl.context, CL_MEM_READ_WRITE, sizeof (arr_b), NULL, NULL);
    cl_mem buf_c = clCreateBuffer (opencl.context, CL_MEM_READ_WRITE, sizeof (arr_c), NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue (opencl.context,
                                                   opencl.default_device,
                                                   0,
                                                   NULL); 
    
    (void)clEnqueueWriteBuffer (queue, buf_a, CL_TRUE, 0, sizeof (arr_a), arr_a, 0, NULL, NULL);
    (void)clEnqueueWriteBuffer (queue, buf_b, CL_TRUE, 0, sizeof (arr_b), arr_b, 0, NULL, NULL);

    const char *kernel_code = 
        "void kernel simple_add(global const int *a, global const int *b, global int *c) {"
        "    int id = get_global_id(0);"
        "    c[id] = a[id] + b[id];"
        "}";

    cl_program program = clCreateProgramWithSource (opencl.context, 1, &kernel_code, NULL, NULL);

    err = clBuildProgram (program, 1, &opencl.default_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        (void)fprintf (stderr, "OpenCL: Error building\n");
        return -1;
    }

    cl_kernel simple_add = clCreateKernel (program, "simple_add", NULL);
    
    (void)clSetKernelArg (simple_add, 0, sizeof (cl_mem), &buf_a);
    (void)clSetKernelArg (simple_add, 1, sizeof (cl_mem), &buf_b);
    (void)clSetKernelArg (simple_add, 2, sizeof (cl_mem), &buf_c);

    (void)clEnqueueNDRangeKernel(queue, simple_add, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    (void)clFinish (queue);

    (void)clEnqueueReadBuffer (queue, buf_c, CL_TRUE, 0, sizeof (arr_c), arr_c, 0, NULL, NULL);


    printf ("  ");
    print_iarr (arr_a, ARRLEN(arr_a));
    printf ("\n");
    printf ("+ ");
    print_iarr (arr_b, ARRLEN(arr_b));
    printf ("\n");
    printf ("----------------------------------------------------------------\n");
    printf ("  ");
    print_iarr (arr_c, ARRLEN(arr_c));
    printf ("\n");

    destroy_cl (&opencl);
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

static int
initialize_cl (cl_setup_t *self)
{
    cl_int err = 0;

    err = get_platforms (&self->all_platforms, &self->num_platforms);
    if (err != 0) { return err; }
    self->default_platform = self->all_platforms[0];
    log_platform_name (self->default_platform, "Using platform");

    err = get_devices (self->default_platform, &self->all_devices, &self->num_devices);
    if (err != 0) { return err; }
    self->default_device = self->all_devices[0];
    log_device_name (self->default_device, "Using device");

    self->context = clCreateContext (NULL, self->num_devices, self->all_devices, NULL, NULL, NULL);
    
    return 0;
}

static void
destroy_cl (cl_setup_t *self)
{
    free (self->all_platforms);
    self->all_platforms = NULL;
    self->num_platforms = 0;

    free (self->all_devices);
    self->all_devices = NULL;
    self->num_devices = 0;
}


static void
log_platform_name (cl_platform_id platform, char *msg)
{
    char *name = NULL;
    size_t name_size = 0;

    /* get size */
    (void)clGetPlatformInfo (platform, CL_PLATFORM_NAME, 0, NULL, &name_size);

    /* allocate buffer */
    name = malloc (name_size);
    assert (name != NULL);

    /* get name */
    (void)clGetPlatformInfo (platform, CL_PLATFORM_NAME, name_size, name, NULL);
    (void)fprintf (stderr, "OpenCL: %s: %s\n", msg, name);

    /* clean up */
    free (name);
    name = NULL;
    name_size = 0;
}

static void
log_device_name (cl_device_id device, char *msg)
{
    char *name = NULL;
    size_t name_size = 0;

    /* get size */
    (void)clGetDeviceInfo (device, CL_DEVICE_NAME, 0, NULL, &name_size);

    /* allocate buffer */
    name = malloc (name_size);
    assert (name != NULL);

    /* get name */
    (void)clGetDeviceInfo (device, CL_DEVICE_NAME, name_size, name, NULL);
    (void)fprintf (stderr, "OpenCL: %s: %s\n", msg, name);

    /* clean up */
    free (name);
    name = NULL;
    name_size = 0;
}


static cl_int
get_platforms (cl_platform_id **p_all_platforms, cl_uint *p_num_platforms)
{
    cl_uint num_platforms = 0;
    cl_platform_id *all_platforms = NULL;

    assert (p_all_platforms != NULL);
    assert (p_num_platforms != NULL);

    /* get platform count */
    (void)clGetPlatformIDs (0, NULL, &num_platforms);
    if (num_platforms == 0)
    {
        (void)fprintf (stderr, "OpenCL: No platforms found.\n");
        return -1;
    }

    /* allocate room for all platforms */
    all_platforms = malloc (num_platforms * sizeof (cl_platform_id));
    assert (all_platforms != NULL);

    /* get platforms */
    (void)clGetPlatformIDs (num_platforms, all_platforms, NULL);

    /* return */
    *p_all_platforms = all_platforms;
    *p_num_platforms = num_platforms;
    return 0;
}


static cl_int
get_devices (cl_platform_id platform, cl_device_id **p_all_devices, cl_uint *p_num_devices)
{
    cl_uint num_devices = 0;
    cl_device_id *all_devices = NULL;

    assert (p_all_devices != NULL);
    assert (p_num_devices != NULL);

    /* get device count */
    (void)clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (num_devices == 0)
    {
        (void)fprintf (stderr, "OpenCL: No devices found.\n");
        return -1;
    }

    /* allocate room for all devices */
    all_devices = malloc (num_devices * sizeof (cl_device_id));
    assert (all_devices != NULL);

    /* get devices */
    (void)clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, num_devices, all_devices, NULL);

    /* return */
    *p_all_devices = all_devices;
    *p_num_devices = num_devices;
    return 0;
}


/* end of file */
