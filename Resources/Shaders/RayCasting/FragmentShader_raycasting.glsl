#version 430

layout(location = 0) in vec3 EntryPoint;
layout(location = 1) in vec4 ExitPointCoord;

uniform sampler2D ExitPoints;
uniform sampler3D VolumeTex;
uniform sampler1D TransferFunc;
uniform float StepSize;
layout(location = 0) out vec4 FragColor;

void main()
{
    vec2 exitFragCoord = (ExitPointCoord.xy / ExitPointCoord.w + 1.0) / 2.0;
    vec3 exitPoint = texture(ExitPoints, exitFragCoord).xyz;
    if (EntryPoint == exitPoint)
        discard;

    vec3 direction = exitPoint - EntryPoint;
    float rayLength = length(direction);
    vec3 deltaDirection = normalize(direction) * StepSize;
    float deltaLength = length(deltaDirection);
    vec3 voxelCoord = EntryPoint;
    vec4 accumulatedColor = vec4(0.0);
    float traversedLength = 0.0;
    vec4 backgroundColor = vec4(1.0, 1.0, 1.0, 0.0);

    for (int i = 0; i < 2000; i++)
    {
        float intensity = texture(VolumeTex, voxelCoord).x;
        vec4 sampledColor = texture(TransferFunc, intensity);

        if (sampledColor.a > 0.0)
        {
            sampledColor.a = 1.0 - pow(1.0 - sampledColor.a, StepSize * 200.0);
            sampledColor.rgb *= sampledColor.a;
            accumulatedColor.rgb += (1.0 - accumulatedColor.a) * sampledColor.rgb;
            accumulatedColor.a += (1.0 - accumulatedColor.a) * sampledColor.a;
        }

        voxelCoord += deltaDirection;
        traversedLength += deltaLength;
        if (traversedLength >= rayLength)
        {
            accumulatedColor.rgb += (1.0 - accumulatedColor.a) * backgroundColor.rgb;
            break;
        }
        if (accumulatedColor.a >= 1.0)
        {
            accumulatedColor.a = 1.0;
            break;
        }
    }

    FragColor = accumulatedColor;
}
