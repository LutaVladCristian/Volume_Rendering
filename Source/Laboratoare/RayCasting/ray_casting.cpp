#include "ray_casting.h"

#include <vector>
#include <iostream>

#include <Core/Engine.h>

using namespace std;


RayCasting::RayCasting()
{
}

RayCasting::~RayCasting()
{
	delete[] volumeData;
	 
}

Mesh* RayCasting::createCube(const char *name)
{
	vector<VertexFormat> vertices
	{
		VertexFormat(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0)),
		VertexFormat(glm::vec3(0, 0, 1), glm::vec3(0, 0, 1)),
		VertexFormat(glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)),
		VertexFormat(glm::vec3(0, 1, 1), glm::vec3(0, 1, 1)),
		VertexFormat(glm::vec3(1, 0, 0), glm::vec3(1, 0, 0)),
		VertexFormat(glm::vec3(1, 0, 1), glm::vec3(1, 0, 1)),
		VertexFormat(glm::vec3(1, 1, 0), glm::vec3(1, 1, 0)),
		VertexFormat(glm::vec3(1, 1, 1), glm::vec3(1, 1, 1))
	
	};
	vector<unsigned int> indices =
	{
		1, 5, 7,
		7, 3, 1,
		0, 2, 6,
		6, 4, 0,
		0, 1, 3,
		3, 2, 0,
		7, 5, 4,
		4, 6, 7,
		2, 3, 7,
		7, 6, 2,
		1, 0, 4,
		4, 5, 1			
	};

	meshes[name] = new Mesh(name);
	meshes[name]->InitFromData(vertices, indices);
	return meshes[name];
}

GLuint RayCasting::createVolumeTexture(const string& fileLocation, unsigned int x, unsigned int y, unsigned int z) {
	
	loadRAWFile(fileLocation, x, y, z);
	
	GLuint g_volTexObj;
	glGenTextures(1, &g_volTexObj);
	// bind 3D texture target
	glBindTexture(GL_TEXTURE_3D, g_volTexObj);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	// pixel transfer happens here from client to OpenGL server
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, xsize, ysize, zsize, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, volumeData);

	cout << "Volume texture created" << endl;
	return g_volTexObj;
}


GLuint RayCasting::createTFTexture(const string& fileLocation) {
	// read in the user defined data of transfer function
	ifstream inFile(fileLocation.c_str(), ifstream::in);
	if (!inFile)
	{
		cerr << "Error openning file: " << fileLocation << endl;
		exit(EXIT_FAILURE);
	}

	const int MAX_CNT = 10000;
	GLubyte *tff = (GLubyte *)calloc(MAX_CNT, sizeof(GLubyte));
	inFile.read(reinterpret_cast<char *>(tff), MAX_CNT);
	if (inFile.eof())
	{
		size_t bytecnt = inFile.gcount();
		*(tff + bytecnt) = '\0';
		cout << "Transfer function read: byte count = " << bytecnt << endl;

	}
	else if (inFile.fail())
	{
		cout << fileLocation << "read failed " << endl;
	}
	else
	{
		cout << fileLocation << "is too large" << endl;
	}
	GLuint tff1DTex;
	glGenTextures(1, &tff1DTex);
	glBindTexture(GL_TEXTURE_1D, tff1DTex);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, tff);
	free(tff);
	return tff1DTex;
}


bool RayCasting::loadRAWFile(const string& fileLocation, unsigned int x, unsigned int y, unsigned int z)
{
	
	FILE *File = NULL;

	if (!fileLocation.c_str())
	{
		cout << fileLocation << "does not exist" << endl;
		return false;
	}

	fopen_s(&File, fileLocation.c_str(), "rb");

	if (!File)
	{
		
		cout << fileLocation << "could not be opened" << endl;
		return false;
	}
		

	xsize = x;
	ysize = y;
	zsize = z;
	volumeData = new unsigned char[xsize * ysize * zsize];

	int sliceSize = xsize*ysize;

	for (int slice = 0; slice < zsize; slice++)
	{
		fread(&volumeData[slice*ysize*xsize], sizeof(unsigned char), sliceSize, File);

	}

	fclose(File);

	return true;

}


void RayCasting::Init()
{
	frameBuffer = new FrameBuffer();
	auto resolution = window->GetResolution();
	frameBuffer->Generate(resolution.x, resolution.y, 3);
	stepSize = 0.001f;
	

	auto camera = GetSceneCamera();
	camera->SetPositionAndRotation(glm::vec3(1, 0.2, 2), glm::quat(glm::vec3(-30 * TO_RADIANS, 45 * TO_RADIANS, 0)));
	camera->Update();

	Mesh *cube = createCube("cube");

	std::string shaderPath = "Source/Laboratoare/RayCasting/Shaders/";

	{
		Shader *shader = new Shader("BackFaceShader");
		shader->AddShader(shaderPath + "VertexShader_backface.glsl", GL_VERTEX_SHADER);
		shader->AddShader(shaderPath + "FragmentShader_backface.glsl", GL_FRAGMENT_SHADER);
		shader->CreateAndLink();
		shaders[shader->GetName()] = shader;
	}


	{
		Shader *shader = new Shader("RayCastingShader");
		shader->AddShader(shaderPath + "VertexShader_raycasting.glsl", GL_VERTEX_SHADER);
		shader->AddShader(shaderPath + "FragmentShader_raycasting.glsl", GL_FRAGMENT_SHADER);
		shader->CreateAndLink();
		shaders[shader->GetName()] = shader;
	}


	volumeTexture = createVolumeTexture(RESOURCE_PATH::VOLUMES + "head256.raw",256,256, 225);
	tfTexture = createTFTexture(RESOURCE_PATH::VOLUMES + "tff.dat");
}

void RayCasting::FrameStart()
{

}

void RayCasting::Update(float deltaTimeSeconds)
{
	frameBuffer->Bind();
	glm::mat4 model_matrix = glm::rotate(glm::mat4(1), -45.f *TO_RADIANS, glm::vec3(0, 1, 0));
	model_matrix = glm::rotate(model_matrix, 90.f * TO_RADIANS, glm::vec3(1,0, 0));
	glm::ivec2 resolution = window->props.resolution;
	
	
	// sets the clear color for the color buffer
	glClearColor(0.5, 0.5, 0.5, 1);
	// clears the color buffer (using the previously set color) and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// sets the screen area where to draw
	glViewport(0, 0, resolution.x, resolution.y);

	{
		auto shader = shaders["BackFaceShader"];
		shader->Use();

		glEnable(GL_CULL_FACE);
		glCullFace(GL_FRONT);
		RenderMesh(meshes["cube"], shaders["BackFaceShader"], model_matrix);
		glDisable(GL_CULL_FACE);
	}

	FrameBuffer::BindDefault();
	{
		glm::ivec2 resolution = window->props.resolution;
		// sets the clear color for the color buffer
		glClearColor(0.2, 0.2, 0.2, 1);
		// clears the color buffer (using the previously set color) and depth buffer
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		// sets the screen area where to draw
		glViewport(0, 0, resolution.x, resolution.y);

		auto shader = shaders["RayCastingShader"];
		shader->Use();
		
		int textureReflexionLoc = shader->GetUniformLocation("ExitPoints");
		glUniform1i(textureReflexionLoc, 0);
		frameBuffer->BindTexture(0, GL_TEXTURE0);


		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_3D, volumeTexture);
		glUniform1i(glGetUniformLocation(shader->program, "VolumeTex"), 1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_1D, tfTexture);
		glUniform1i(glGetUniformLocation(shader->program, "TransferFunc"), 2);

		glUniform1f(glGetUniformLocation(shader->program, "StepSize"), stepSize);
		meshes["cube"]->UseMaterials(false);
		RenderMesh(meshes["cube"], shaders["RayCastingShader"], model_matrix);
	}
}

void RayCasting::FrameEnd()
{
	
}

void RayCasting::OnInputUpdate(float deltaTime, int mods)
{
	
};

void RayCasting::OnKeyPress(int key, int mods)
{

};

void RayCasting::OnKeyRelease(int key, int mods)
{
	// add key release event
};

void RayCasting::OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY)
{
	// add mouse move event
};

void RayCasting::OnMouseBtnPress(int mouseX, int mouseY, int button, int mods)
{
	// add mouse button press event
};

void RayCasting::OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods)
{
	// add mouse button release event
}

void RayCasting::OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY)
{
	// treat mouse scroll event
}

void RayCasting::OnWindowResize(int width, int height)
{
	// treat window resize event
	frameBuffer->Generate(width, height, 3);
}
