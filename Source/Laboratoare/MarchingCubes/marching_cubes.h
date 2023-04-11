#pragma once

#include <Component/SimpleScene.h>

using namespace std;


typedef struct {
	glm::vec3 p[8]; //pozitie
	glm::vec3 n[8]; //normala
	double val[8];  //densitatea tesutului

} GRIDCELL;

typedef struct {
	glm::vec3 p[3]; //varfurile
	glm::vec3 c;	//centrul
	glm::vec3 n[3]; //normale
} TRIANGLE;


class MarchingCubes : public SimpleScene
{
	
	public:
		MarchingCubes();
		~MarchingCubes();

		void Init() override;


		bool loadRAWFile(const string& fileLocation, unsigned int xsize, unsigned int ysize, unsigned int zsize);
		glm::vec3 VertexInterp(double isolevel, glm::vec3 p1, glm::vec3 p2, double valp1, double valp2);
		glm::vec3 VertexNormal(int i, int j, int k);
		int PolygoniseCube(GRIDCELL g, double iso, TRIANGLE *tri);
		void reconstructSurface(Mesh *mesh);

	private:

		void FrameStart() override;
		void Update(float deltaTimeSeconds) override;
		void FrameEnd() override;

		void OnInputUpdate(float deltaTime, int mods) override;
		void OnKeyPress(int key, int mods) override;
		void OnKeyRelease(int key, int mods) override;
		void OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY) override;
		void OnMouseBtnPress(int mouseX, int mouseY, int button, int mods) override;
		void OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods) override;
		void OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY) override;
		void OnWindowResize(int width, int height) override;

	private:
		unsigned char *volumeData;
		unsigned int xsize, ysize, zsize;
		double isolevel;
		TRIANGLE *tri;

};
