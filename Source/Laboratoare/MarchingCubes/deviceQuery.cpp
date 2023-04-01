#include "deviceQuery.h"


// Timer variables
float pcFreq;
long long startMoment;
float hostTime, devTime;

void cleanup()
{
	cudaDeviceReset();
}

// Verifica daca exista eroare CUDA
void checkCUDAError(const char* msg)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		getchar();
		exit(EXIT_FAILURE);
	}
}

bool initCUDA(void)
{
#if __DEVICE_EMULATION__
	return true;
#else
	int driverVersion = 0, runtimeVersion = 0;
	int count = 0;
	int i = 0;

	SAFE_CUDA_CALL(cudaGetDeviceCount(&count));
	if (count == 0)
	{
		fprintf(stderr, "Nu exista nici un device.\n");
		return false;
	}

	printf("Exista %d device-uri capabile de CUDA.\n", count);

	for (i = 0; i < count; i++)
	{
		cudaDeviceProp prop;
		SAFE_CUDA_CALL(cudaGetDeviceProperties(&prop, i));
		{
			if (prop.major >= 1)
			{
				break;
			}
		}
	}
	if (i == count)
	{
		fprintf(stderr, "Nu exista nici un device care suporta CUDA.\n");
		return false;
	}

	cudaDeviceProp deviceProp;
	// Uncomment to use maxflops device, but cuda_helper functions need to be used
	//int devID = gpuGetMaxGflopsDeviceId();

	for (int dev = 0; dev < count; dev++)
	{
		int devID = dev;
		SAFE_CUDA_CALL(cudaSetDevice(devID));
		SAFE_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// Driver version and compute capability
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  Device %d: \"%s\" - CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			devID, deviceProp.name, driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  Device %d: \"%s\" - CUDA Capability Major/Minor version number:    %d.%d\n",
			devID, deviceProp.name, deviceProp.major, deviceProp.minor);

		// Memory amount
		char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(msg, sizeof(msg),
			"Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#else
		snprintf(msg, sizeof(msg),
			"Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#endif
		printf("  Device %d: \"%s\" - %s", devID, deviceProp.name, msg);

		// SMs and SP (Cuda cores)
		printf("  Device %d: \"%s\" - (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
			devID, deviceProp.name, deviceProp.multiProcessorCount,
			getSPcoresPerSM(deviceProp),
			getSPcoresPerSM(deviceProp) *
			deviceProp.multiProcessorCount);

		// GPU/CPU Clock Freq
		printf("  Device %d: \"%s\" - GPU Max Clock rate:                             %.0f MHz (%0.2f "
			"GHz)\n",
			devID, deviceProp.name, deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

		// Memory Freq
#if CUDART_VERSION >= 5000
// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Device %d: \"%s\" - Memory Clock rate:				   %.0f Mhz\n",
			devID, deviceProp.name, deviceProp.memoryClockRate * 1e-3f);
		printf("  Device %d: \"%s\" -Memory Bus Width:				   %d-bit\n",
			devID, deviceProp.name, deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  Device %d: \"%s\" -L2 Cache Size:				   %d bytes\n",
				devID, deviceProp.name, deviceProp.l2CacheSize);
		}
#endif

		// Other device properties
		printf("  Device %d: \"%s\" - Total amount of constant memory:               %zu bytes\n",
			devID, deviceProp.name, deviceProp.totalConstMem);
		printf("  Device %d: \"%s\" - Total amount of shared memory per block:       %zu bytes\n",
			devID, deviceProp.name, deviceProp.sharedMemPerBlock);
		printf("  Device %d: \"%s\" - Total shared memory per multiprocessor:        %zu bytes\n",
			devID, deviceProp.name, deviceProp.sharedMemPerMultiprocessor);
		printf("  Device %d: \"%s\" - Total number of registers available per block: %d\n",
			devID, deviceProp.name, deviceProp.regsPerBlock);
		printf("  Device %d: \"%s\" - Warp size:                                     %d\n",
			devID, deviceProp.name, deviceProp.warpSize);
		printf("  Device %d: \"%s\" - Maximum number of threads per multiprocessor:  %d\n",
			devID, deviceProp.name, deviceProp.maxThreadsPerMultiProcessor);
		printf("  Device %d: \"%s\" - Maximum number of threads per block:           %d\n",
			devID, deviceProp.name, deviceProp.maxThreadsPerBlock);
		printf("  Device %d: \"%s\" - Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			devID, deviceProp.name, deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Device %d: \"%s\" - Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			devID, deviceProp.name, deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Device %d: \"%s\" - Maximum memory pitch:                          %zu bytes\n",
			devID, deviceProp.name, deviceProp.memPitch);
		printf("  Device %d: \"%s\" - Texture alignment:                             %zu bytes\n",
			devID, deviceProp.name, deviceProp.textureAlignment);
		printf(
			"  Device %d: \"%s\" - Concurrent copy and kernel execution:          %s with %d copy "
			"engine(s)\n",
			devID, deviceProp.name, (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Device %d: \"%s\" - Run time limit on kernels:                     %s\n",
			devID, deviceProp.name, deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Device %d: \"%s\" - Integrated GPU sharing Host Memory:            %s\n",
			devID, deviceProp.name, deviceProp.integrated ? "Yes" : "No");
		printf("  Device %d: \"%s\" - Support host page-locked memory mapping:       %s\n",
			devID, deviceProp.name, deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Device %d: \"%s\" - Alignment requirement for Surfaces:            %s\n",
			devID, deviceProp.name, deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device %d: \"%s\" - Device has ECC support:                        %s\n",
			devID, deviceProp.name, deviceProp.ECCEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  Device %d: \"%s\" - CUDA Device Driver Mode (TCC or WDDM):         %s\n",
			devID, deviceProp.name, deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
			: "WDDM (Windows Display Driver Model)");
#endif

		printf("  Device %d: \"%s\" - Device supports Unified Addressing (UVA):      %s\n",
			devID, deviceProp.name, deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device %d: \"%s\" - Device supports Managed Memory:                %s\n",
			devID, deviceProp.name, deviceProp.managedMemory ? "Yes" : "No");
		printf("  Device %d: \"%s\" - Device supports Compute Preemption:            %s\n",
			devID, deviceProp.name, deviceProp.computePreemptionSupported ? "Yes" : "No");
		printf("  Device %d: \"%s\" - Supports Cooperative Kernel Launch:            %s\n",
			devID, deviceProp.name, deviceProp.cooperativeLaunch ? "Yes" : "No");
		printf("  Device %d: \"%s\" - Supports MultiDevice Co-op Kernel Launch:      %s\n",
			devID, deviceProp.name, deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		printf("  Device %d: \"%s\" - Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
			devID, deviceProp.name, deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char* sComputeMode[] = {
		"Default (multiple host threads can use ::cudaSetDevice() with device "
		"simultaneously)",
		"Exclusive (only one host thread in one process is able to use "
		"::cudaSetDevice() with this device)",
		"Prohibited (no host thread can use ::cudaSetDevice() with this "
		"device)",
		"Exclusive Process (many threads in one process is able to use "
		"::cudaSetDevice() with this device)",
		"Unknown",
		NULL };
		printf("  Device %d: \"%s\" - Compute Mode:\n", devID, deviceProp.name);
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);

	}

	printf("CUDA initializat cu succes\n");

	return true;
#endif
}