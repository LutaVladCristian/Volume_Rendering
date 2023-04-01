#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <Windows.h>	// For the performance counters
#endif

#include "Utils.h"

static int bRandomSeedApplied = 0;	// Marks if the seed for the random generator has been set

// ===========================================================
// Use the first available CUDA device and read its properties
// ===========================================================

void getFirstDeviceProperties(struct cudaDeviceProp * devProps)
{
	int devCount = 0;

	memset(devProps, 0, sizeof(struct cudaDeviceProp));

	// Get the number of available CUDA devices
	SAFE_CUDA_CALL(cudaGetDeviceCount(&devCount));
	VALIDATE(devCount > 0, "Error: No CUDA-capable device found.");
	
	// Use the first available device
	SAFE_CUDA_CALL(cudaSetDevice(0));

	// Get the properties of this device
	SAFE_CUDA_CALL(cudaGetDeviceProperties(devProps, 0));
}

// ==============================================================
// Get number of SP (Cuda Cores) per SM based on the architecture
// ==============================================================

int getSPcoresPerSM(cudaDeviceProp devProp)
{
	int cores = 0;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1) cores = 48;
		else cores = 32;
		break;
	case 3: // Kepler
		cores =192;
		break;
	case 5: // Maxwell
		cores = 128;
		break;
	case 6: // Pascal
		if ((devProp.minor == 1) || (devProp.minor == 2)) cores = 128;
		else if (devProp.minor == 0) cores = 64;
		else printf("Unknown device type\n");
		break;
	case 7: // Volta and Turing
		if ((devProp.minor == 0) || (devProp.minor == 5)) cores =64;
		else printf("Unknown device type\n");
		break;
	case 8: // Ampere
		if (devProp.minor == 0) cores = 64;
		else if (devProp.minor == 6) cores = 128;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

// ====================================================================================================================================
// Validate an array of command-line arguments
// Order of arguments:
// argv[1] - number of dimensions for the global (dimGrid) and local (dimBlock) parameters
// argv[2] .. argv[argv[1] + 1] - global data size on each dimension (optional, if globalSize != NULL use the size from that variable)
// argv[argv[1] + 2] .. argv[2 * argv[1] + 1] - number of local threads of a block (blockSize) on each dimension
// ===================================================================================================================================
void validateArguments(const int argCount, const int expectedMinArgCount, 
	char ** argValues, int * noDimensions, dim3 * vValues, const char ** vErrMessages, 
	dim3 *globalSize)
{
	struct cudaDeviceProp devProps;

	VALIDATE(argCount - 1 >= expectedMinArgCount, "Error: The number of arguments is too small.");

	// Number of dimensions
	*noDimensions = atoi(argValues[1]);
	VALIDATE(*noDimensions > 0 && *noDimensions <= 3, "Error: The number of dimensions is incorrect.");

	// Process the tuples for the dimensions (depending on globalSize != NULL) 
	if(globalSize != NULL)
		VALIDATE(argCount - 2 >= *noDimensions, "Error: The number of arguments is different than expected.");
	else
		VALIDATE(argCount - 2 >= 2 * (*noDimensions), "Error: The number of arguments is different than expected.");

	// By convention:
	// - vValues[0] holds the number of blocks, which must be calculated (dimGrid) per dimension
	// - vValues[1] holds the number of threads per block (dimBlock) per dimension
	// - vValues[2] holds the TOTAL size of the grid (total threads) per dimension
	int i = 2;

	// Read the total data size to process(if it is not given from the command line) per dimension
	// and the threads per block per dimension
	int j = 2;
	if (globalSize != NULL)
		vValues[j--] = *globalSize;
	for (int k = j; k >= 1; k--)
	{
		vValues[k].x = atoi(argValues[i++]);
		if (*noDimensions >= 2)
			vValues[k].y = atoi(argValues[i++]);
		if (*noDimensions == 3)
			vValues[k].z = atoi(argValues[i++]);
	}
	VALIDATE((int)vValues[2].x > 0 && (int)vValues[2].y > 0 && (int)vValues[2].z > 0, vErrMessages[1]);
	VALIDATE((int)vValues[1].x > 0 && (int)vValues[1].y > 0 && (int)vValues[1].z > 0, vErrMessages[0]);

	// Extract the CUDA device properties
	getFirstDeviceProperties(&devProps);

	// Ensure the number of threads per block is valid
	VALIDATE((int)vValues[1].x * (int)vValues[1].y * (int)vValues[1].z <= devProps.maxThreadsPerBlock, "Error: Too many threads per block.");
	// Ensure that the blockSize per dimensions is valid
	VALIDATE((int)vValues[1].x  <= devProps.maxThreadsDim[0], "Error: Too many threads per block.x .");
	VALIDATE((int)vValues[1].y  <= devProps.maxThreadsDim[1], "Error: Too many threads per block.y .");
	VALIDATE((int)vValues[1].z  <= devProps.maxThreadsDim[2], "Error: Too many threads per block.z .");

	// Compute the number of needed blocks
	vValues[0].x = (vValues[2].x + vValues[1].x - 1) / vValues[1].x;
	if (*noDimensions >= 2)
		vValues[0].y = (vValues[2].y + vValues[1].y - 1) / vValues[1].y;
	if (*noDimensions >= 3)
		vValues[0].z = (vValues[2].z + vValues[1].z - 1) / vValues[1].z;

	// Ensure that the number of blocks is valid
	VALIDATE((int)vValues[0].x <= devProps.maxGridSize[0], "Error: Too many blocks.");
	VALIDATE((int)vValues[0].y <= devProps.maxGridSize[1], "Error: Too many blocks.");
	VALIDATE((int)vValues[0].z <= devProps.maxGridSize[2], "Error: Too many blocks.");
}

// =============================================
// Create and optionally fill a host-based array
// =============================================

void generateHostData(int vSize, float ** vData, int mustFill)
{
	// Allocate memory for the array
	* vData = (float *)malloc(vSize * sizeof(float));

	// Ensure the array has been properly allocated
	VALIDATE((* vData) != NULL, "Error: Could not allocate enough memory.");

	// Filling the array is optional
	if (mustFill)
	{
		// Initialize the random generator seed, if necessary
		if (!bRandomSeedApplied)
		{
			bRandomSeedApplied = 1;
			srand((unsigned int)time(NULL));
		}

		// The generated numbers are between 0 and 1
		for (int i = 0; i < vSize; ++i)
		{
			(* vData)[i] = (float)rand() / RAND_MAX;
		}
	}
}

// =================================================
// Allocate and optionally fill a device-based array
// =================================================
void generateDeviceData(int byteSize, float ** vDevData, const float * vHostData, int copyFromHost)
{
	// First we allocate device memory for the device array
	SAFE_CUDA_CALL(cudaMalloc((void **)vDevData, byteSize));

	// If necessary, we also fill the device-based array
	if (copyFromHost && (vHostData != NULL))
	{
		SAFE_CUDA_CALL(cudaMemcpy(* vDevData, vHostData, byteSize, cudaMemcpyHostToDevice));
	}
}

// ================================
// Compare the contents of 2 arrays
// ================================

void compareResults(const float * h_a, const float * h_b, int size)
{
	int match = TRUE;

	for (int i = 0; i < size; ++i)
	{
		if (fabs(h_a[i] - h_b[i]) > 1.0E-5F)
		{
			match = FALSE;
			break;
		}
	}

	printf("Match? %s" NEW_LINE, match? "Yes" : "No");
}

// =====================================
// Measuring time accurately on the host
// =====================================

void hostTimerStart(float * pcFreq, long long * startMoment)
{
#ifdef _WIN32

	LARGE_INTEGER pcParam;

    VALIDATE(QueryPerformanceFrequency(&pcParam), "Error: QueryPerformanceFrequency has failed.");

	// Get frequency for microsecond timer
    * pcFreq = (float)pcParam.QuadPart * 1.0E-6F;

	// Get the starting moment
    QueryPerformanceCounter(&pcParam);
    * startMoment = pcParam.QuadPart;

#endif
}

float hostTimerStop(const float pcFreq, long long startMoment)
{
#ifdef _WIN32

	LARGE_INTEGER pcParam;

	// Get the current moment
    QueryPerformanceCounter(&pcParam);

	// Calculate the time difference
    return (float)(pcParam.QuadPart - startMoment) / pcFreq;

#else

	// Change this if running on a non-Windows OS
	return 0.0F;

#endif
}