#pragma once

#include <Component/SimpleScene.h>
using namespace std;


class RayCasting : public SimpleScene
{
	

	public:
		RayCasting();
		~RayCasting();

		void Init() override;
		bool loadRAWFile(const string& fileLocation, unsigned int x, unsigned int y, unsigned int z);
		Mesh *createCube(const char *name);
		GLuint createVolumeTexture(const string& fileLocation, unsigned int x, unsigned int y, unsigned int z);
		GLuint createTFTexture(const string& fileLocation);
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
		FrameBuffer *frameBuffer;
		GLuint volumeTexture;
		GLuint tfTexture;
		float stepSize;

};
