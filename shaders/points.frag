#version 450 core
in vec4 v_color;
out vec4 frag;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0; // [-1,1]^2
    float r2 = dot(p,p);
    if (r2 > 1.0) discard;              // circular sprite
    float edge = smoothstep(1.0, 0.7, r2); // feather
    frag = v_color * edge;
}
