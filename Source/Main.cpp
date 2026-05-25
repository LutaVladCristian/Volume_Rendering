#include <ctime>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace std;

#include <Core/Engine.h>

#include <Laboratoare/LabList.h>

namespace
{
	void PrintUsage()
	{
		cout << "Usage:\n"
			<< "  VolumeRendering.exe [--ray-casting [raw width height depth [transfer-function]]]\n"
			<< "  VolumeRendering.exe --marching-cubes [raw width height depth [isovalue]]\n";
	}

	unsigned int ParseDimension(const char* value)
	{
		auto parsed = stoul(value);
		if (parsed == 0)
			throw invalid_argument("Volume dimensions must be greater than zero.");
		return static_cast<unsigned int>(parsed);
	}
}

int main(int argc, char **argv)
{
	srand((unsigned int)time(NULL));

	bool useMarchingCubes = false;
	string volumeFile;
	string transferFunctionFile = "Resources/Volumes/tff.dat";
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int depth = 0;
	double isolevel = 50.0;

	try
	{
		if (argc > 1)
		{
			string sceneArgument = argv[1];
			if (sceneArgument == "--marching-cubes")
			{
				useMarchingCubes = true;
				volumeFile = "Resources/Volumes/Bucky.raw";
				width = height = depth = 32;
				if (argc != 2 && argc != 6 && argc != 7)
					throw invalid_argument("Invalid Marching Cubes arguments.");
				if (argc >= 6)
				{
					volumeFile = argv[2];
					width = ParseDimension(argv[3]);
					height = ParseDimension(argv[4]);
					depth = ParseDimension(argv[5]);
				}
				if (argc == 7)
					isolevel = stod(argv[6]);
			}
			else if (sceneArgument == "--ray-casting")
			{
				volumeFile = "Resources/Volumes/head256.raw";
				width = 256;
				height = 256;
				depth = 225;
				if (argc != 2 && argc != 6 && argc != 7)
					throw invalid_argument("Invalid ray-casting arguments.");
				if (argc >= 6)
				{
					volumeFile = argv[2];
					width = ParseDimension(argv[3]);
					height = ParseDimension(argv[4]);
					depth = ParseDimension(argv[5]);
				}
				if (argc == 7)
					transferFunctionFile = argv[6];
			}
			else
			{
				throw invalid_argument("Unknown scene argument.");
			}
		}
		else
		{
			volumeFile = "Resources/Volumes/head256.raw";
			width = 256;
			height = 256;
			depth = 225;
		}
	}
	catch (const exception& error)
	{
		cerr << error.what() << endl;
		PrintUsage();
		return 1;
	}

	// Create a window property structure
	WindowProperties wp;
	wp.resolution = glm::ivec2(1280, 720);

	// Init the Engine and create a new window with the defined properties
	Engine::Init(wp);

	try
	{
		unique_ptr<World> world;
		if (useMarchingCubes)
			world.reset(new MarchingCubes(volumeFile, width, height, depth, isolevel));
		else
			world.reset(new RayCasting(volumeFile, width, height, depth, transferFunctionFile));

		world->Init();
		world->Run();
	}
	catch (const exception& error)
	{
		cerr << error.what() << endl;
		Engine::Exit();
		return 1;
	}

	// Signals to the Engine to release the OpenGL context
	Engine::Exit();

	return 0;
}
