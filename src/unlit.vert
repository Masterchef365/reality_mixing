#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_multiview : require

layout(binding = 0) uniform Animation {
    mat4 camera[2];
    float anim;
    float ppx;
    float ppy;
    float fx;
    float fy;
    float coeffs[5];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

vec3 deproject(in vec3 pixel);

void main() {
    gl_Position = camera[gl_ViewIndex] * vec4(deproject(inPosition) / 1000., 1.0);
    gl_PointSize = 2.;
    fragColor = inColor;
}

vec3 deproject(in vec3 pixel) {
    vec2 xy = (pixel.xy - vec2(ppx, ppy)) / vec2(fx, fy);
    float r2 = dot(xy, xy);
    float f = 1 + coeffs[0]*r2 + coeffs[1]*r2*r2 + coeffs[4]*r2*r2*r2;
    vec2 uv = xy * f + 2.*vec2(coeffs[2], coeffs[3]) * xy.x * xy.y + (r2 + 2 * xy.x * xy.y) * vec2(coeffs[3], coeffs[2]);
    return pixel.z * vec3(-uv, 1.);
}

/*
layout(push_constant) uniform Model {
    mat4 model;
};
gl_Position = camera[gl_ViewIndex] * model * vec4(inPosition, 1.0);
*/
