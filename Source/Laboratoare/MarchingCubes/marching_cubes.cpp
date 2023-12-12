#include "marching_cubes.h"
#include "LookupTables.h"

#include <vector>
#include <iostream>

#include <Core/Engine.h>

using namespace std;

float epsilon = 0.000001;

MarchingCubes::MarchingCubes()
{
}

MarchingCubes::~MarchingCubes()
{
	//distruge obiecte

	if (volumeData) delete[] volumeData;
	if (tri) delete[] tri;
}

bool MarchingCubes::loadRAWFile(const string& fileLocation, unsigned int x, unsigned int y, unsigned int z) //functie pentru a incarca datele de volum
{
	
	FILE *File = NULL;

	if (!fileLocation.c_str())
	{
		return false;
	}

	fopen_s(&File, fileLocation.c_str(), "rb");

	if (!File)
		return false;

	xsize = x;
	ysize = y;
	zsize = z;
	volumeData = new unsigned char[xsize * ysize * zsize]; //se aloca un string de caractere de pana la 8 biti

	int sliceSize = xsize*ysize;

	for (int slice = 0; slice < zsize; slice++) //se face segmentarea de-a lungul axei OZ 
	{
		fread(&volumeData[slice*ysize*xsize], sizeof(unsigned char), sliceSize, File);

	}

	fclose(File);

	return true;

}

glm::vec3 MarchingCubes::VertexInterp(double isolevel, glm::vec3 p1, glm::vec3 p2, double valp1, double valp2)
{
	double mu;
	glm::vec3 p;

	if (abs(isolevel - valp1) < epsilon)
		return(p1);
	if (abs(isolevel - valp2) < epsilon)
		return(p2);
	if (abs(valp1 - valp2) < epsilon)
		return(p1);


	mu = (isolevel - valp1) / (valp2 - valp1);
	p.x = p1.x + mu * (p2.x - p1.x);
	p.y = p1.y + mu * (p2.y - p1.y);
	p.z = p1.z + mu * (p2.z - p1.z);

	return(p);
}


int MarchingCubes::PolygoniseCube(GRIDCELL g, double iso, TRIANGLE *tri) {

	int i, ntri = 0;
	int cubeindex;
	glm::vec3 vertlist[12];
	glm::vec3 normlist[12];
	glm::vec3 normal1, normal2, normal3;

	/*
	Determine the index into the edge table which
	tells us which vertices are inside of the surface
	*/
	cubeindex = 0;
	if (g.val[0] < iso) cubeindex |= 1;
	if (g.val[1] < iso) cubeindex |= 2;
	if (g.val[2] < iso) cubeindex |= 4;
	if (g.val[3] < iso) cubeindex |= 8;
	if (g.val[4] < iso) cubeindex |= 16;
	if (g.val[5] < iso) cubeindex |= 32;
	if (g.val[6] < iso) cubeindex |= 64;
	if (g.val[7] < iso) cubeindex |= 128;

	/* Cube is entirely in/out of the surface */
	if (edgeTable[cubeindex] == 0)
		return(0);

	/* Find the vertices where the surface intersects the cube */
	if (edgeTable[cubeindex] & 1) {
		vertlist[0] = VertexInterp(iso, g.p[0], g.p[1], g.val[0], g.val[1]);
		normlist[0] = VertexInterp(iso, g.n[0], g.n[1], g.val[0], g.val[1]);
	}
	if (edgeTable[cubeindex] & 2) {
		vertlist[1] = VertexInterp(iso, g.p[1], g.p[2], g.val[1], g.val[2]);
		normlist[1] = VertexInterp(iso, g.n[1], g.n[2], g.val[1], g.val[2]);
	}
	if (edgeTable[cubeindex] & 4) {
		vertlist[2] = VertexInterp(iso, g.p[2], g.p[3], g.val[2], g.val[3]);
		normlist[2] = VertexInterp(iso, g.n[2], g.n[3], g.val[2], g.val[3]);
	}
	if (edgeTable[cubeindex] & 8) {
		vertlist[3] = VertexInterp(iso, g.p[3], g.p[0], g.val[3], g.val[0]);
		normlist[3] = VertexInterp(iso, g.n[3], g.n[0], g.val[3], g.val[0]);
	}
	if (edgeTable[cubeindex] & 16) {
		vertlist[4] = VertexInterp(iso, g.p[4], g.p[5], g.val[4], g.val[5]);
		normlist[4] = VertexInterp(iso, g.n[4], g.n[5], g.val[4], g.val[5]);
	}
	if (edgeTable[cubeindex] & 32) {
		vertlist[5] = VertexInterp(iso, g.p[5], g.p[6], g.val[5], g.val[6]);
		normlist[5] = VertexInterp(iso, g.n[5], g.n[6], g.val[5], g.val[6]);
	}
	if (edgeTable[cubeindex] & 64) {
		vertlist[6] = VertexInterp(iso, g.p[6], g.p[7], g.val[6], g.val[7]);
		normlist[6] = VertexInterp(iso, g.n[6], g.n[7], g.val[6], g.val[7]);
	}
	if (edgeTable[cubeindex] & 128) {
		vertlist[7] = VertexInterp(iso, g.p[7], g.p[4], g.val[7], g.val[4]);
		normlist[7] = VertexInterp(iso, g.n[7], g.n[4], g.val[7], g.val[4]);
	}
	if (edgeTable[cubeindex] & 256) {
		vertlist[8] = VertexInterp(iso, g.p[0], g.p[4], g.val[0], g.val[4]);
		normlist[8] = VertexInterp(iso, g.n[0], g.n[4], g.val[0], g.val[4]);
	}
	if (edgeTable[cubeindex] & 512) {
		vertlist[9] = VertexInterp(iso, g.p[1], g.p[5], g.val[1], g.val[5]);
		normlist[9] = VertexInterp(iso, g.n[1], g.n[5], g.val[1], g.val[5]);
	}
	if (edgeTable[cubeindex] & 1024) {
		vertlist[10] = VertexInterp(iso, g.p[2], g.p[6], g.val[2], g.val[6]);
		normlist[10] = VertexInterp(iso, g.n[2], g.n[6], g.val[2], g.val[6]);
	}
	if (edgeTable[cubeindex] & 2048) {
		vertlist[11] = VertexInterp(iso, g.p[3], g.p[7], g.val[3], g.val[7]);
		normlist[11] = VertexInterp(iso, g.n[3], g.n[7], g.val[3], g.val[7]);
	}

	/* Create the triangles */
	for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
		tri[ntri].p[0] = vertlist[triTable[cubeindex][i]];
		tri[ntri].p[1] = vertlist[triTable[cubeindex][i + 1]];
		tri[ntri].p[2] = vertlist[triTable[cubeindex][i + 2]];

		normal1 = normlist[triTable[cubeindex][i]];
		normal2 = normlist[triTable[cubeindex][i + 1]];
		normal3 = normlist[triTable[cubeindex][i + 2]];

		//TODO - calculeaza normala pe fiecare varf, ca interpolare de gradienti		
		if (glm::length(normal1) > epsilon)
			normal1 = glm::normalize(normal1);

		if (glm::length(normal2) > epsilon)
			normal2 = glm::normalize(normal2);

		if (glm::length(normal3) > epsilon)
			normal3 = glm::normalize(normal3);
		
		tri[ntri].n[0] = normal1;
		tri[ntri].n[1] = normal2;
		tri[ntri].n[2] = normal3;

		ntri++;
	}

	return(ntri);

}


glm::vec3 MarchingCubes::VertexNormal(int i, int j, int k)
{

	glm::vec3 normal(0);

	if (i - 1 < 0)
		normal = normal + glm::vec3(volumeData[k * xsize * ysize + j * xsize + i + 1] - volumeData[k * xsize * ysize + j * xsize + 0], 0, 0);
	if (i + 1 > xsize)
		normal = normal + glm::vec3(volumeData[k * xsize * ysize + j * xsize + xsize] - volumeData[k * xsize * ysize + j * xsize + i - 1], 0, 0);
	if ((i - 1 > 0) && (i + 1 < xsize))
		normal = normal + glm::vec3(volumeData[k * xsize * ysize + j * xsize + i + 1] - volumeData[k * xsize * ysize + j * xsize + i - 1], 0, 0);


	if (j - 1 < 0)
		normal = normal + glm::vec3(0, volumeData[k * xsize * ysize + (j + 1) * xsize + i] - volumeData[k * xsize * ysize + 0 * xsize + i], 0);
	if (j + 1 > ysize)
		normal = normal + glm::vec3(0, volumeData[k * xsize * ysize + ysize * xsize + i] - volumeData[k * xsize * ysize + (j - 1) * xsize + i], 0);
	if ((j - 1 > 0)&&(j + 1 < ysize))
		normal = normal + glm::vec3(0, volumeData[k * xsize * ysize + (j + 1) * xsize + i] - volumeData[k * xsize * ysize + (j - 1) * xsize + i], 0);


	if (k - 1 < 0)
		normal = normal + glm::vec3(0, 0, volumeData[(k + 1) * xsize * ysize + j * xsize + i] - volumeData[0 * xsize * ysize + j * xsize + i]);
	if (k + 1 > zsize)
		normal = normal + glm::vec3(0, 0, volumeData[zsize * xsize * ysize + j * xsize + i] - volumeData[(k - 1) * xsize * ysize + j * xsize + i]);
	if ((k - 1 > 0) && (k + 1 < zsize))
		normal = normal + glm::vec3(0, 0, volumeData[(k + 1) * xsize * ysize + j * xsize + i] - volumeData[(k - 1) * xsize * ysize + j * xsize + i]);

	return normal;
}




void MarchingCubes::reconstructSurface(Mesh *mesh)
{
	GRIDCELL grid;
	TRIANGLE triangles[10];

	int ntri = 0;
	
	int i, j, k, n, l;

	glm::vec3 normal;
	for (i = 0; i < xsize - 1; i++)
	{
		for (j = 0; j < ysize - 1; j++)
		{
			for (k = 0; k < zsize - 1; k++)
			{
				//TODO - completeaza informatiile din voxelul "grid" in functie de "volumeData"
				
				//p[0] cu(i, j, k)
				grid.p[0] = glm::vec3(i,j,k); 
				grid.val[0] = volumeData[k*xsize*ysize + j*xsize + i];
				grid.n[0] = VertexNormal(i,j,k);
				
				//p[1] cu(i + 1, j, k)
				grid.p[1] = glm::vec3(i+1, j, k);
				grid.val[1] = volumeData[k * xsize * ysize + j * xsize + i + 1];
				grid.n[1] = VertexNormal(i+1, j, k);


				//p[2] cu (i+1,j+1,k)
				grid.p[2] = glm::vec3(i + 1, j + 1, k);
				grid.val[2] = volumeData[k * xsize * ysize +( j + 1) * xsize + i + 1];
				grid.n[2] = VertexNormal(i+1, j+1, k);

				//p[3] cu (i,j+1,k)
				grid.p[3] = glm::vec3(i, j + 1, k);
				grid.val[3] = volumeData[k * xsize * ysize + (j + 1) * xsize + i];
				grid.n[3] = VertexNormal(i, j+1, k);

				//p[4] cu (i,j,k+1)
				grid.p[4] = glm::vec3(i, j, k+1);
				grid.val[4] = volumeData[(k+1) * xsize * ysize + j * xsize + i];
				grid.n[4] = VertexNormal(i, j, k+1);

				//p[5] cu (i+1,j,k+1)
				grid.p[5] = glm::vec3(i + 1, j, k + 1);
				grid.val[5] = volumeData[(k+1) * xsize * ysize + j * xsize + i + 1];
				grid.n[5] = VertexNormal(i+1, j, k+1);

				//p[6] cu (i+1,j+1,k+1)
				grid.p[6] = glm::vec3(i + 1, j + 1, k + 1);
				grid.val[6] = volumeData[(k + 1) * xsize * ysize + (j + 1) * xsize + i + 1];
				grid.n[6] = VertexNormal(i+1, j+1, k+1);

				//p[7] cu (i,j+1,k+1)
				grid.p[7] = glm::vec3(i, j + 1, k + 1);
				grid.val[7] = volumeData[(k + 1) * xsize * ysize + (j + 1)* xsize + i];
				grid.n[7] = VertexNormal(i, j+1, k+1);

				n = PolygoniseCube(grid,isolevel,triangles);
				
				tri = (TRIANGLE*)realloc(tri, (ntri + n) * sizeof(TRIANGLE));
				for (l = 0; l<n; l++)
					tri[ntri + l] = triangles[l];
				ntri += n;
			}
		}
	}
	
	vector<glm::vec3> positions;
	vector<glm::vec3> normals;
	vector<unsigned int> indices;
	for (i = 0; i < ntri; i++)
	{
		positions.push_back(glm::vec3(tri[i].p[0].x, tri[i].p[0].y, tri[i].p[0].z));
		positions.push_back(glm::vec3(tri[i].p[1].x, tri[i].p[1].y, tri[i].p[1].z));
		positions.push_back(glm::vec3(tri[i].p[2].x, tri[i].p[2].y, tri[i].p[2].z));
		
		normals.push_back(glm::vec3(tri[i].n[0].x, tri[i].n[0].y, tri[i].n[0].z));
		normals.push_back(glm::vec3(tri[i].n[1].x, tri[i].n[1].y, tri[i].n[1].z));
		normals.push_back(glm::vec3(tri[i].n[2].x, tri[i].n[2].y, tri[i].n[2].z));

		indices.push_back(i * 3);
		indices.push_back(i * 3 + 1);
		indices.push_back(i * 3 + 2);
	}
	
	mesh->InitFromData(positions, normals, indices);
}


void MarchingCubes::Init()
{
	tri = NULL;

	auto camera = GetSceneCamera();
	camera->SetPositionAndRotation(glm::vec3(2, 5, 10), glm::quat(glm::vec3(-30 * TO_RADIANS, 0, 0)));
	camera->Update();

	// Load a mesh from file into GPU memory
	{
		loadRAWFile(RESOURCE_PATH::VOLUMES + "Bucky.raw", 32, 32, 32);
		isolevel = 50;
		Mesh* mesh_volume = new Mesh("volume");
		reconstructSurface(mesh_volume);
		meshes[mesh_volume->GetMeshID()] = mesh_volume;
	}

	std::string shaderPath = "Source/Laboratoare/MarchingCubes/Shaders/";

	// Create a shader program for rendering to texture
	{
		Shader *shader = new Shader("MainShader");
		shader->AddShader(shaderPath + "VertexShader.glsl", GL_VERTEX_SHADER);

		shader->AddShader(shaderPath + "FragmentShader.glsl", GL_FRAGMENT_SHADER);
		shader->CreateAndLink();
		shaders[shader->GetName()] = shader;
	}
}

void MarchingCubes::FrameStart()
{

}

void MarchingCubes::Update(float deltaTimeSeconds)
{
	ClearScreen();

	{
		auto shader = shaders["MainShader"];

		shader->Use();

		RenderMesh(meshes["volume"], shaders["MainShader"], glm::vec3(0, 0, 0), glm::vec3(5.0f/xsize));
	}
}

void MarchingCubes::FrameEnd()
{
	//DrawCoordinatSystem();
}

void MarchingCubes::OnInputUpdate(float deltaTime, int mods)
{

	//TODO schimba isovaloarea
	if (window->KeyHold(GLFW_KEY_Z) && isolevel < 200)
	{

		isolevel += 100 * deltaTime;
		cout << "Iso value:" << isolevel << endl;
		reconstructSurface(meshes["volume"]);
	}
		
	if (window->KeyHold(GLFW_KEY_X) && isolevel > 5)
	{
		isolevel -= 100 * deltaTime;
		cout << "Iso value:" << isolevel << endl;
		reconstructSurface(meshes["volume"]);
	}
};

void MarchingCubes::OnKeyPress(int key, int mods)
{
	


};

void MarchingCubes::OnKeyRelease(int key, int mods)
{
	// add key release event
};

void MarchingCubes::OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY)
{
	// add mouse move event
};

void MarchingCubes::OnMouseBtnPress(int mouseX, int mouseY, int button, int mods)
{
	// add mouse button press event
};

void MarchingCubes::OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods)
{
	// add mouse button release event
}

void MarchingCubes::OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY)
{
	// treat mouse scroll event
}

void MarchingCubes::OnWindowResize(int width, int height)
{
	// treat window resize event
}
