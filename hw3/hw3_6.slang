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

// Find the closest triangle intersection using BVH
bool intersect_mesh_bvh(
    float3 origin,
    float3 direction,
    TriangleMesh mesh,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<uint3> triangles,
    StructuredBuffer<BVHNode> bvh_nodes,
    float t_min,
    float t_max,
    out float hit_t,
    out int hit_tri,
    out float3 hit_bary) {

    hit_t = t_max;
    hit_tri = -1;
    hit_bary = float3(0.0, 0.0, 0.0);

    // Start at the mesh's BVH root
    int node_idx = int(mesh.bvh_root_index);
    int end_idx = bvh_nodes[node_idx].next_index;

    // Stackless BVH traversal
    while (node_idx < end_idx) {
        BVHNode node = bvh_nodes[node_idx];

        // Skip the subtree if the ray misses its box
        if (!intersect_aabb(
                node.bbox_min,
                node.bbox_max,
                origin,
                direction,
                t_min,
                hit_t)) {

            node_idx = node.next_index;
            continue;
        }

        // If leaf, test its triangles
        if (node.is_leaf) {
            for (uint j = node.triangle_begin;
                 j < node.triangle_end; j++) {

                uint3 tri = triangles[j]
                          + mesh.vertex_offset;

                float3 p0 = vertices[tri.x];
                float3 p1 = vertices[tri.y];
                float3 p2 = vertices[tri.z];

                float3 bary;
                float t;

                if (intersect_tri(
                        p0, p1, p2,
                        origin, direction,
                        t_min, hit_t,
                        bary, t)) {

                    hit_t = t;
                    hit_tri = int(j);
                    hit_bary = bary;
                }
            }
        }

        // Continue to the next node in preorder
        node_idx++;
    }

    return hit_tri >= 0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<TriangleMesh> meshes,
                  StructuredBuffer<float3> vertices,
                  StructuredBuffer<float3> vertex_normals,
                  StructuredBuffer<uint3> triangles,
                  StructuredBuffer<BVHNode> bvh_nodes,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {

    uint width, height;
    image.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // Generate camera ray
    float aspect = float(width) / float(height);
    float u = (float(tid.x) + 0.5) / float(width);
    float v = (float(tid.y) + 0.5) / float(height);

    float half_width = tan(radians(camera.fov_x) / 2.0);
    float half_height = half_width / aspect;

    // Transform ray from camera space to world space
    float3 origin = xform_pt(
        camera.cam_to_world,
        float3(0.0, 0.0, 0.0)
    );

    float3 direction = xform_vec(
        camera.cam_to_world,
        float3(
            (2.0 * u - 1.0) * half_width,
            (1.0 - 2.0 * v) * half_height,
            -1.0
        )
    );

    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);

    // Trace camera and reflected rays
    for (int bounce = 0; bounce < max_depth; bounce++) {
        float closest_t = 1e30;
        int closest_shape = -1;
        int closest_tri = -1;
        float3 closest_bary = float3(0.0, 0.0, 0.0);

        // Find the closest intersection
        for (int i = 0; i < num_shapes; i++) {
            Shape shape = shapes[i];

            // Transform ray into object space
            float3 obj_origin = xform_pt(
                shape.world_to_obj, origin
            );

            float3 obj_direction = xform_vec(
                shape.world_to_obj, direction
            );

            // Sphere intersection
            if (shape.type == ShapeType.Sphere) {
                Sphere sphere = spheres[shape.index];

                float t;
                if (intersect_sphere(
                        obj_origin, obj_direction,
                        sphere, EPSILON, closest_t, t)) {

                    closest_t = t;
                    closest_shape = i;
                    closest_tri = -1;
                }
            }

            // Triangle mesh intersection using BVH
            else if (shape.type == ShapeType.TriangleMesh) {
                TriangleMesh mesh = meshes[shape.index];

                float t;
                int tri_idx;
                float3 bary;

                if (intersect_mesh_bvh(
                        obj_origin,
                        obj_direction,
                        mesh,
                        vertices,
                        triangles,
                        bvh_nodes,
                        EPSILON,
                        closest_t,
                        t,
                        tri_idx,
                        bary)) {

                    closest_t = t;
                    closest_shape = i;
                    closest_tri = tri_idx;
                    closest_bary = bary;
                }
            }
        }

        // If ray misses, add background
        if (closest_shape < 0) {
            color += throughput * background;
            break;
        }

        Shape shape = shapes[closest_shape];

        // Calculate object-space intersection
        float3 obj_origin = xform_pt(
            shape.world_to_obj, origin
        );

        float3 obj_direction = xform_vec(
            shape.world_to_obj, direction
        );

        float3 obj_hit_point =
            obj_origin + closest_t * obj_direction;

        // Transform intersection to world space
        float3 hit_point = xform_pt(
            shape.obj_to_world, obj_hit_point
        );

        float3 obj_normal = float3(0.0, 0.0, 0.0);

        // Calculate sphere normal
        if (shape.type == ShapeType.Sphere) {
            Sphere sphere = spheres[shape.index];

            obj_normal = normalize(
                obj_hit_point - sphere.center
            );
        }

        // Calculate triangle mesh normal
        else if (shape.type == ShapeType.TriangleMesh) {
            TriangleMesh mesh = meshes[shape.index];

            uint3 tri = triangles[uint(closest_tri)]
                      + mesh.vertex_offset;

            // Get vertex normals
            float3 n0 = vertex_normals[tri.x];
            float3 n1 = vertex_normals[tri.y];
            float3 n2 = vertex_normals[tri.z];

            // Interpolate using barycentric coordinates
            obj_normal = normalize(
                closest_bary.x * n0 +
                closest_bary.y * n1 +
                closest_bary.z * n2
            );
        }

        // Transform normal using inverse transpose
        float3 normal = normalize(
            xform_vec(
                transpose(shape.world_to_obj),
                obj_normal
            )
        );

        float3 diffuse_color = float3(0.0, 0.0, 0.0);

        // Calculate diffuse lighting
        for (int j = 0; j < num_lights; j++) {
            PointLight light = lights[j];

            float3 to_light = light.position - hit_point;
            float light_dist = length(to_light);
            float3 light_dir = to_light / light_dist;

            float diffuse = max(
                dot(normal, light_dir), 0.0
            );

            if (diffuse <= 0.0) {
                continue;
            }

            // Check for shadows
            bool in_shadow = false;

            for (int k = 0; k < num_shapes; k++) {
                Shape shadow_shape = shapes[k];

                // Transform shadow ray into object space
                float3 shadow_origin = xform_pt(
                    shadow_shape.world_to_obj, hit_point
                );

                float3 shadow_direction = xform_vec(
                    shadow_shape.world_to_obj, light_dir
                );

                // Sphere shadow intersection
                if (shadow_shape.type == ShapeType.Sphere) {
                    Sphere shadow_sphere =
                        spheres[shadow_shape.index];

                    float shadow_t;

                    if (intersect_sphere(
                            shadow_origin,
                            shadow_direction,
                            shadow_sphere,
                            EPSILON, light_dist,
                            shadow_t)) {

                        in_shadow = true;
                        break;
                    }
                }

                // Triangle mesh shadow intersection using BVH
                else if (shadow_shape.type ==
                         ShapeType.TriangleMesh) {

                    TriangleMesh mesh =
                        meshes[shadow_shape.index];

                    float shadow_t;
                    int shadow_tri;
                    float3 shadow_bary;

                    if (intersect_mesh_bvh(
                            shadow_origin,
                            shadow_direction,
                            mesh,
                            vertices,
                            triangles,
                            bvh_nodes,
                            EPSILON,
                            light_dist,
                            shadow_t,
                            shadow_tri,
                            shadow_bary)) {

                        in_shadow = true;
                        break;
                    }
                }
            }

            // Add lighting if not in shadow
            if (!in_shadow) {
                diffuse_color +=
                    shape.Kd * light.intensity *
                    diffuse / (light_dist * light_dist);
            }
        }

        // Add diffuse color from this bounce
        color += throughput * diffuse_color;

        // Update reflection strength
        throughput *= shape.Kr;

        // Stop if material is not reflective
        if (max(throughput.x,
                max(throughput.y, throughput.z)) <= 0.0) {
            break;
        }

        // Reflect ray for the next bounce
        origin = hit_point;
        direction = normalize(
            reflect(direction, normal)
        );
    }

    image[tid] = float4(color, 1.0);
}
