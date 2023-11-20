
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "device_functions.h"

#include <chrono>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include <iostream>

#include "stb_image.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define M_PI 3.14159265358979323846

const char *blurHashForFile(int xComponents, int yComponents,const char *filename);

const char *blurHashForPixels(int xComponents, int yComponents, int width, int height, uint8_t *rgb, size_t bytesPerRow);
const char *blurHashForPixelsGPU(int xComponents, int yComponents, int width, int height, uint8_t *rgb, size_t bytesPerRow);
static float *multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t *rgb, size_t bytesPerRow);
static char *encode_int(int value, int length, char *destination);

static int encodeDC(float r, float g, float b);
static int encodeAC(float r, float g, float b, float maximumValue);


static inline int linearTosRGB(float value);
static inline float sRGBToLinear(int value);
static inline float signPow(float value, float exp);

//sRGBToLinear na urzadzeniu
__device__ static float dev_sRGBToLinear(int value) {
	float v = (float)value / 255;
	if(v <= 0.04045) return v / 12.92;
	else return powf((v + 0.055) / 1.055, 2.4);
	
}

//zrownoleglone multiplyBasisFunction
__global__ void multiplyBasisFunctionDev1(int xcomponents, int ycomponents, int width, int height, uint8_t* rgb, size_t bytesPerRow, float* factors, int cycles, int thrno)
{
	int yComponent = blockIdx.x/xcomponents, xComponent = blockIdx.x%xcomponents;
	int cidx = threadIdx.x;
	
	float basis;
	int x,y;
	float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;
	float scale = normalisation / (width * height);
	float r = 0, g = 0, b = 0;
	for(int i = 0;i<cycles; i++)
	{
		if (cidx < width*height)
		{
			y = cidx / width;
			x = cidx % width;
			basis =  cosf(M_PI * xComponent * x / width) *  cosf(M_PI * yComponent * y / height);
			r += basis * dev_sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]);
			g += basis * dev_sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]);
			b += basis * dev_sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]);
		}
		cidx += thrno;
	}
	atomicAdd(&(factors[yComponent*3*xcomponents + 3*xComponent + 0]),r*scale);
	atomicAdd(&(factors[yComponent*3*xcomponents + 3*xComponent + 1]),g*scale);
	atomicAdd(&(factors[yComponent*3*xcomponents + 3*xComponent + 2]),b*scale);
}

int main(int argc, char** argv)

{
	

	if(argc != 4) {
		fprintf(stderr, "Usage: %s x_components y_components imagefile\n", argv[0]);
		return 1;
	}

	int xComponents = atoi(argv[1]);
	int yComponents = atoi(argv[2]);
	if(xComponents < 1 || xComponents > 9 || yComponents < 1 || yComponents > 9) {
		fprintf(stderr, "incorrect usage parameters, Usage: %s x_components y_components imagefile\n", argv[0]);
		return 1;
	}


	//cpu compute
	const char *hash = blurHashForFile(xComponents, yComponents, argv[3]);

	if(!hash) {
		fprintf(stderr, "Failed to load image file \"%s\", usage: %s x_components y_components imagefile\n", argv[3], argv[0]);
		return 1;
	}


	// gpu compute
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
    gpuErrchk(cudaStatus);

	int width, height, channels;
	unsigned char *data = stbi_load(argv[3], &width, &height, &channels, 3);
	if(!data) return NULL;

	printf("%s\n", hash);
	fflush(stdout);

	const char *gpuHash = blurHashForPixelsGPU(xComponents, yComponents, width, height, data, width * 3);

	stbi_image_free(data);



	//printf("%s\n", hash);
	printf("%s\n", gpuHash);
	fflush(stdout);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


const char *blurHashForPixelsGPU(int xComponents, int yComponents, int width, int height, uint8_t *rgb, size_t bytesPerRow) {
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];
	cudaError_t cudaStatus;

	//error check
	if(xComponents < 1 || xComponents > 9) return NULL;
	if(yComponents < 1 || yComponents > 9) return NULL;

	float *factors = (float*)calloc(1, sizeof(float)*xComponents * yComponents * 3);
	float *dev_factors=0;
	uint8_t *dev_rgb =0;

	float milliseconds;
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	cudaStatus = cudaMalloc((void**)&dev_rgb, sizeof(uint8_t) * height * width * 3 );
	gpuErrchk(cudaStatus);

	cudaStatus = cudaMalloc((void**)&dev_factors, sizeof(float)*xComponents * yComponents * 3);
    gpuErrchk(cudaStatus);

	cudaStatus = cudaMemset(dev_factors, 0, sizeof(float)*xComponents * yComponents * 3);
    gpuErrchk(cudaStatus);

	cudaStatus = cudaMemcpy(dev_rgb, rgb, sizeof(uint8_t) * height * width * 3, cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);

	cudaDeviceProp cdp;
	cudaStatus = cudaGetDeviceProperties(&cdp,0);
	int thrno = cdp.maxThreadsPerBlock >= 1024 ? 1024 : 512;
    gpuErrchk( cudaGetDeviceProperties(&cdp,0) );
	int cycles  = ((width*height) / thrno) + ((width*height) % thrno == 0 ? 0 : 1);
	cudaEventRecord(start);
	//oblicznie dla komponentu DC kazdej pary komponentow AC,jeden blok na pare komponentow AC/ komponent DC
	multiplyBasisFunctionDev1<<<(yComponents*xComponents),thrno>>>(xComponents, yComponents, width, height, dev_rgb, bytesPerRow, dev_factors, cycles, thrno);
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"gpu processed in: "<<milliseconds<<" ms\n";

	cudaStatus = cudaMemcpy(factors, dev_factors, sizeof(float) * xComponents * yComponents * 3, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaStatus);


	float *dc = factors;
	float *ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char *ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	//encode no of componets and max ac component value
	if(acCount > 0) {
		float actualMaximumValue = 0;
		for(int i = 0; i < acCount * 3; i++) {
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	} else {
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}
	//encode avg collor
	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);
	//encode components
	for(int i = 0; i < acCount; i++) {
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	//deallocations

	cudaFree(dev_rgb);
	cudaFree(dev_factors);
	free(factors);

	*ptr = 0;

	return buffer;
}


const char *blurHashForPixels(int xComponents, int yComponents, int width, int height, uint8_t *rgb, size_t bytesPerRow) {
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];


	//error check
	if(xComponents < 1 || xComponents > 9) return NULL;
	if(yComponents < 1 || yComponents > 9) return NULL;

	

	//allocations

	float *factors = (float*)calloc(1, sizeof(float)*xComponents * yComponents * 3);

	//copying data
	auto p1 = std::chrono::high_resolution_clock::now();
	for(int y = 0; y < yComponents; y++) {
		for(int x = 0; x < xComponents; x++) {
			float *factor = multiplyBasisFunction(x, y, width, height, rgb, bytesPerRow);
			factors[ y*3*xComponents + 3*x] = factor[0];
			factors[ y*3*xComponents + 3*x + 1] = factor[1];
			factors[ y*3*xComponents + 3*x + 2] = factor[2];
		}
	}
	auto p2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> proc_double = p2 - p1;
	std::cout <<"locally processed in: "<< proc_double.count() << " ms\n";
	/*for(int y = 0; y < yComponents; y++) {
		for(int x = 0; x < xComponents; x++) {
			std::cout << factors[y*3*xComponents + 3*x]<<"|"<<factors[y*3*xComponents + 3*x +1]<<"|"<<factors[y*3*xComponents + 3*x +2] <<" ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/

	float *dc = factors;
	float *ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char *ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	//encode no of componets and max ac component value
	if(acCount > 0) {
		float actualMaximumValue = 0;
		for(int i = 0; i < acCount * 3; i++) {
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	} else {
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}
	//encode avg collor
	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);
	//encode components
	for(int i = 0; i < acCount; i++) {
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}
	free(factors);

	*ptr = 0;

	return buffer;
}


static float *multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t *rgb, size_t bytesPerRow) {
	float r = 0, g = 0, b = 0;
	float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			float basis = cosf(M_PI * xComponent * x / width) * cosf(M_PI * yComponent * y / height);
			r += basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]);
			g += basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]);
			b += basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]);
		}
	}

	float scale = normalisation / (width * height);

	static float result[3];
	result[0] = r * scale;
	result[1] = g * scale;
	result[2] = b * scale;
	//printf("%f\n", scale);
	/*result[0] = r ;
	result[1] = g ;
	result[2] = b ;*/
	return result;
}



static int encodeDC(float r, float g, float b) {
	int roundedR = linearTosRGB(r);
	int roundedG = linearTosRGB(g);
	int roundedB = linearTosRGB(b);
	return (roundedR << 16) + (roundedG << 8) + roundedB;
}

static int encodeAC(float r, float g, float b, float maximumValue) {
	int quantR = fmaxf(0, fminf(18, floorf(signPow(r / maximumValue, 0.5) * 9 + 9.5)));
	int quantG = fmaxf(0, fminf(18, floorf(signPow(g / maximumValue, 0.5) * 9 + 9.5)));
	int quantB = fmaxf(0, fminf(18, floorf(signPow(b / maximumValue, 0.5) * 9 + 9.5)));

	return quantR * 19 * 19 + quantG * 19 + quantB;
}

const char characters[83]={'0','1','2','3','4','5','6','7','8','9',
'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
'#','$','%','*','+',',','-','.',':',';','=','?','@','[',']','^','_','{','|','}','~'};

static char *encode_int(int value, int length, char *destination) {
	int divisor = 1;
	for(int i = 0; i < length - 1; i++) divisor *= 83;

	for(int i = 0; i < length; i++) {
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}

static inline int linearTosRGB(float value) {
	float v = fmaxf(0, fminf(1, value));
	if(v <= 0.0031308) return v * 12.92 * 255 + 0.5;
	else return (1.055 * powf(v, 1 / 2.4) - 0.055) * 255 + 0.5;
}

static inline float sRGBToLinear(int value) {
	float v = (float)value / 255;
	if(v <= 0.04045) return v / 12.92;
	else return powf((v + 0.055) / 1.055, 2.4);
	
}

static inline float signPow(float value, float exp) {
	return copysignf(powf(fabsf(value), exp), value);
}

const char *blurHashForFile(int xComponents, int yComponents,const char *filename) {
	int width, height, channels;
	unsigned char *data = stbi_load(filename, &width, &height, &channels, 3);
	if(!data) return NULL;
	const char *hash = blurHashForPixels(xComponents, yComponents, width, height, data, width * 3);

	stbi_image_free(data);

	return hash;
}