#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

struct Vertex {
	vec3 pos;
	vec4 color;
	vec3 normal;
	vec2 uv;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer {
	Vertex vertices[];
};

layout( push_constant ) uniform constants {
	mat4 tr;
	VertexBuffer vbuf;
};

void main() {
	Vertex v = vbuf.vertices[gl_VertexIndex];
	gl_Position = tr * vec4(v.pos, 1.0f);
	outColor = v.color.xyz;
	outUV = v.uv;
}
