#version 430

layout(location = 0) in vec3 frag_color;

layout(location = 0) out vec4 out_color;

void main()
{
	//out_color = vec4(frag_color,0);
	out_color = vec4(1, 0, 0, 0);
}