#version 450 core
layout(location=0) in vec4 a_pos_size; // xyz = pos, w = size (px)
layout(location=1) in vec4 a_color;

uniform mat4 uViewProj;

out vec4 v_color;

void main() {
    gl_Position = uViewProj * vec4(a_pos_size.xyz, 1.0);
    gl_PointSize = a_pos_size.w;
    v_color = a_color;
}
