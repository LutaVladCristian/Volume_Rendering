#version 430

layout (location = 0) in vec3 EntryPoint;
layout (location = 1) in vec4 ExitPointCoord;

uniform sampler2D ExitPoints;
uniform sampler3D VolumeTex;
uniform sampler1D TransferFunc;  
uniform float     StepSize;
layout (location = 0) out vec4 FragColor;

void main()
{
    vec2 exitFragCoord = vec2(2); //TODO calculeaza coordonatele pentru accesarea texturii ExitPoints in functie de ExitPointCoord (componentele x,y si w)
    vec3 exitPoint = texture(ExitPoints, exitFragCoord).xyz;    //TODO acceseaza coordonatele texturii ExitPoints in functie de coordonatele de textura
    if (EntryPoint == exitPoint) 
    	//daca sunt pe background nu e nevoie sa calculez nimic
    	discard;
    vec3 dir = exitPoint - EntryPoint;	//TODO calculeaza directia razei ca punctul de iesire minus punctul de intrare
    float len = 0;// lungimea razei e calculata pentru terminarea razei
    vec3 deltaDir = vec3(0);	//TODO calculeaza vectorul delta cu care se inainteaza pe raza, in functie de StepSize si de directia razei
    float deltaDirLen = StepSize;	
    vec3 voxelCoord = EntryPoint; 
    vec4 colorAcum = vec4(0.0);		// culoarea acumulata (Cdst)
    float alphaAcum = 0.0;          // opacitatea acumulata (Adst)
    float intensity;
    float lengthAcum = 0.0;
    vec4 colorSample;				// culoarea curenta (Csrc) 
    float alphaSample;				// opacitatea curenta (Asrc)
    
    vec4 bgColor = vec4(1.0, 1.0, 1.0, 0.0);	//culoarea de fundal
 
    for(int i = 0; i < 2000; i++)	//nu calculez mai mult de 2000 culori per raza
    {
    	intensity =  0;				//TODO determinati intensitatea in functie de textura VolumeTex si de coordonatele voxelului curent
									//vedeti ca textura intoarce un vec4 si intensity este un float (luati primul canal)
    	
		colorSample = vec4(0); //TODO determinati culoarea cu functia de transfer (in functie de textura TransferFunc si intensitatea voxelului)
    	
    	
    	if (colorSample.a > 0.0) {
            // se moduleaza opacitatea si culoarea lui colorSample dupa niste formule
            colorSample.a = 1.0 - pow(1.0 - colorSample.a, StepSize * 200.0f);
            colorSample.rgb *= colorSample.a;

            // compunere din fata in spate
            colorAcum.rgb += (1.0 - colorAcum.a) * colorSample.rgb;
            colorAcum.a += (1.0 - colorAcum.a) * colorSample.a;
			
    	}
    	//TODO actualizeaza pozitia lui voxelCoord

    	//TODO actualizeaza lungimea parcursa pe raza
		
        if (lengthAcum >= len )	
    	{	
    	    colorAcum.rgb = colorAcum.rgb + (1 - colorAcum.a)*bgColor.rgb;		
    	    break;  // s-a terminat daca am iesit cu raza din volum	
    	}	
    	else if (colorAcum.a > 1.0)
    	{
    	    colorAcum.a = 1.0;
    	    break;	//s-a terminat daca opacitatea acumulata a ajuns la 1
    	}
    }
    FragColor = colorAcum;
    // for test
    // FragColor = vec4(EntryPoint, 1.0);
    // FragColor = vec4(exitPoint, 1.0);
   
}
