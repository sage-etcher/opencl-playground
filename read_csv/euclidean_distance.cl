
__kernel void distance(float16 pos, __global float16 *points, __global float *dist, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n)
    {
        return;
    }

    float16 diff = pow((pos - points[gid]), 2);
    float sum = 0;

    __attribute__((opencl_unroll_hint(16)))
    for (unsigned int i = 0; i < 16; i++)
    {
        if (pos[i] != 0)
        {
            sum += diff[i];
        }
    }

    float euclidian_dist = sqrt(sum);

    dist[gid] = euclidian_dist;
}
