#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#include <iostream>
#include <ctime>


// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

// Helper CUDA
//#include <helper_cuda.h>
//#include <helper_functions.h> 

// =============
// Useful macros
// =============

#define NEW_LINE "\r\n"

// Wait for input from the user, then exit the application
#define WAIT_AND_EXIT(exit_code)	do { system("pause"); exit(exit_code); } while (0)

// CUDA call guard with error signaling
#define SAFE_CUDA_CALL(call)		do {											\
	cudaError_t status = (call);													\
	if (status != cudaSuccess)	{													\
		fprintf(stderr, "Call '%s' at '%s':%d failed with error: '%s'" NEW_LINE,	\
			#call, __FILE__, __LINE__, cudaGetErrorString(status));					\
		WAIT_AND_EXIT(1);															\
	} } while (0)

// Parameter validation macro
#define VALIDATE(cond, err_msg)		do {			\
	if (!(cond)) {									\
		fprintf(stderr, "%s%s", err_msg, NEW_LINE);	\
		WAIT_AND_EXIT(1);							\
	} } while (0)

#define SAFE_PTR_SET(ptr, value)	do { if (ptr) (* ptr) = (value); } while (0)

#define SAFE_FREE(ptr)				do { if (ptr) free(ptr); } while (0)

#define CHECK_VALUE(value, status)	do { if (value) { printf("Condition match: \"%s\" (%s:%d)"	\
														NEW_LINE, #value, __FILE__, __LINE__);	\
													  return (status); } } while (0)

#define TRUE	1
#define FALSE	0

// =================
// Support functions
// =================

// Get SP cores per SM
int getSPcoresPerSM(cudaDeviceProp devProp);

// Validate an array of command-line arguments
void validateArguments(const int argCount, const int expectedArgCount, 
	char ** argValues, int * noDimensions, dim3 * vValues, const char ** vErrMessages, dim3 *globalSize);

// Create and optionally fill a host-based array
void generateHostData(int vSize, float ** vData, int mustFill);
void generateDeviceData(int byteSize, float ** vDevData, const float * vHostData, int copyFromHost);
void compareResults(const float * h_a, const float * h_b, int size);

// Accurate time measurements on the host
void hostTimerStart(float * pcFreq, long long * startMoment);
float hostTimerStop(const float pcFreq, long long startMoment);

#endif	// _COMMON_H_