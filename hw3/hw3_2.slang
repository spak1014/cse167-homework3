import hw3_common;

// Find the closest valid intersection with a sphere
bool intersect_sphere(float3 origin, float3 direction,
                    Sphere sphere, float t_min, float t_max,
                    out float hit_t) {
    float3 oc = origin - sphere.center;

    float a = dot(direction, direction);
    float b = 2.0 * dot(oc, direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    float2 roots;
    hit_t = t_max;

    if (!solve_quadratic(a, b, c, roots)) {
        return false;
    }

    float t1 = min(roots.x, roots.y);
    float t2 = max(roots.x, roots.y);

    if (t1 > t_min && t1 < t_max) {
        hit_t = t1;
        return true;
    }

    if (t2 > t_min && t2 < t_max) {
        hit_t = t2;
        return true;
    }

    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
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
    int closest_shape = -1;

    // Find the closest sphere
    for (int i = 0; i < num_shapes; i++) {
        Sphere sphere = spheres[shapes[i].index];

        float t;
        if (intersect_sphere(origin, direction, sphere,
                            EPSILON, closest_t, t)) {
            closest_t = t;
            closest_shape = i;
        }
    }

    float3 color = background;

    if (closest_shape >= 0) {
        Shape shape = shapes[closest_shape];
        Sphere sphere = spheres[shape.index];

        float3 hit_point = origin + closest_t * direction;
        float3 normal = normalize(hit_point - sphere.center);

        color = float3(0.0, 0.0, 0.0);

        // Calculate lighting from each point light
        for (int j = 0; j < num_lights; j++) {
            PointLight light = lights[j];

            float3 to_light = light.position - hit_point;
            float light_dist = length(to_light);
            float3 light_dir = to_light / light_dist;

            float diffuse = max(dot(normal, light_dir), 0.0);

            if (diffuse <= 0.0) {
                continue;
            }

            // Check whether another sphere blocks the light
            bool in_shadow = false;

            for (int k = 0; k < num_shapes; k++) {
                Sphere shadow_sphere = spheres[shapes[k].index];

                float shadow_t;
                if (intersect_sphere(hit_point, light_dir,
                                    shadow_sphere,
                                    EPSILON, light_dist,
                                    shadow_t)) {
                    in_shadow = true;
                    break;
                }
            }

            // Add diffuse lighting if not in shadow
            if (!in_shadow) {
                float3 contribution =
                    shape.Kd * light.intensity *
                    diffuse / (light_dist * light_dist);

                color += contribution;
            }
        }
    }

    image[tid] = float4(color, 1.0);
}
