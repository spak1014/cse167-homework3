import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    // Generate a ray through the pixel center
    float aspect = float(width) / float(height); 
    float u = (float(tid.x) + 0.5) / float(width); 
    float v = (float(tid.y) + 0.5) / float(height); 

    float3 origin = float3(0.0, 0.0, 0.0); 
    float3 direction = normalize(float3(
        (2.0 * u - 1.0) * aspect, 
        1.0 - 2.0 * v,
        -1.0
    )); 

    float closest_t = 1e30; 
    float3 color = background; 

    // Find the closest sphere intersection
    for (int i = 0; i < num_spheres; i++) {
        Sphere sphere = spheres[i]; 

        float3 oc = origin - sphere.center; 
        float a = dot(direction, direction); 
        float b = 2.0 * dot(oc, direction); 
        float c = dot(oc, oc) - sphere.radius * sphere.radius; 

        float discriminant = b * b - 4.0 * a * c; 

        if (discriminant >= 0.0) {
            float sqrt_d = sqrt(discriminant); 
            float t1 = (-b - sqrt_d) / (2.0 * a); 
            float t2 = (-b + sqrt_d) / (2.0 * a); 

            float t = t1; 
            if (t <= 0.0) {
                t = t2; 
            }

            if (t > 0.0 && t < closest_t) {
                closest_t = t; 

                float3 hit_point = origin + t * direction; 
                float3 normal = normalize(hit_point - sphere.center); 

                color = (normal + float3(1.0, 1.0, 1.0)) * 0.5; 
            }
        }

    }

    image[tid] = float4(color, 1.0);    
}
