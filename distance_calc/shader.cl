
__kernel void distance(float3 pos, __global float3 *points, __global float *dist, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n)
    {
        return;
    }
    
    float3 diff = pow((pos - points[gid]), 2);
    float  hypo = sqrt(diff.s0 + diff.s1 + diff.s2);

    dist[gid] = hypo;
}