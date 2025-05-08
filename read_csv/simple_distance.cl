
__kernel void distance(float16 pos, __global float16 *points, __global float *dist, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n)
    {
        return;
    }
    
    float16 diff = fabs(pos - points[gid]);
    float sum = 0;

    __attribute__((opencl_unroll_hint(16)))
    for (int i = 16; i > 0; i--)
    {
        sum += diff[i];
    }

    dist[gid] = sum;
}