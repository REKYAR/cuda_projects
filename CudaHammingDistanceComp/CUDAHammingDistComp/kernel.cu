#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//#include <device_functions.h>
#include "device_functions.h"

#include <stdio.h>
#include <bitset>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ unsigned long long int dev_counter=0;

__global__ void CompareWordsV3(unsigned int* dev_arr, int strno, int arrlen, int threadno, bool display_pairs, bool* global_indecies ){
    int tidx = blockIdx.x * threadno + threadIdx.x;
    int result;
    unsigned long long pair_count=0;
    unsigned long long cnt=0;
    if (tidx % 10000 == 0)
    {
        printf("block %d - thread %d\n", blockIdx.x, tidx);
    }
    if (tidx < strno)
    {
        for (size_t i = tidx + 1; i < strno; i++)
        {
            for (size_t j = 0; j < arrlen; j++)
            {
                result = dev_arr[strno * j + tidx] ^ dev_arr[strno * j + i];
                cnt += __popc(result);
                if (cnt>1)
                {
                    break;
                }
            }
            if (cnt==1)
            {
                if (display_pairs)
                {
                    global_indecies[tidx * strno + i] = true;
                    pair_count++;
                }else{
                    pair_count++;
                    //printf("+");
                }
            }
            cnt=0;
            __syncthreads();
        }
    }
    if (pair_count != 0)
    {
        atomicAdd(&dev_counter, pair_count);
    }
    
}

int numberOfSetBits(uint32_t i)
{
     i = i - ((i >> 1) & 0x55555555);        // add pairs of bits
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);  // quads
     i = (i + (i >> 4)) & 0x0F0F0F0F;        // groups of 8
     return (i * 0x01010101) >> 24;          // horizontal sum of bytes
}
    
 int compareWordsLocal(unsigned int* dev_arr, int strno, int arrlen, int threadno, bool display_pairs, bool* global_indecies){
        int tidx;
        unsigned int result;
        unsigned long long pair_count=0;
        unsigned int cnt=0;
    for (size_t i = 0; i < strno; i++)
    {
        tidx=i;
        if (tidx % 10000 == 0)
        {
            printf("iteration %d\n", tidx);
        }
        for (size_t i = tidx + 1; i < strno; i++)
        {
            for (size_t j = 0; j < arrlen; j++)
            {
                //result = dev_arr[arrlen * tidx + j] ^ dev_arr[arrlen * i + j];
                //standard indexation
                result = dev_arr[strno * j + tidx] ^ dev_arr[strno * j + i];
                while (result != 0)
                {
                    cnt += result & 1;
                    result>>=1;
                }
                //cnt += numberOfSetBits(result); 
                if (cnt>1)
                {
                    break;
                }
            }
            if (cnt==1)
            {
                if (display_pairs)
                {
                    global_indecies[tidx * strno + i] = true;
                    pair_count++;
                }else{
                    pair_count++;
                }
            }
            cnt=0;
        }
    }
    return pair_count;
}

int main(int argc,char** argv)
{
    bool display = false;
    bool local = false;
    std::string path;
    if (argc == 1)
    {
        return 1;
    }
    for (size_t i = 0; i < argc; i++)
    {
        if(i == 1){
            path = argv[i];
        }else if( strcmp(argv[i], "-d")==0){
            display = true;
        }else if (strcmp(argv[i], "-l")==0){
            local = true;
        }
    }


    unsigned long long counter = 0;
    std::vector<std::string> strings;
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int intsize = 8 * sizeof(unsigned int);
    std::ifstream inputFile;
    inputFile.open(path);
    cudaError_t cudaStatus;
    std::string inStr;
    int strno, strlen, arrlen;
    char cc;
    auto l1 = std::chrono::high_resolution_clock::now();
    inputFile >> strno >>cc>> strlen;
    inputFile.close();
    arrlen = strlen/intsize + (strlen % intsize == 0 ? 0 : 1);
    cudaDeviceProp cdp;

    cudaStatus = cudaSetDevice(0);
    gpuErrchk(cudaStatus);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    //    return 1;
    //}
    if (display)
    {
        strings = std::vector<std::string>(strno);
    }
    
    cudaStatus = cudaGetDeviceProperties(&cdp,0);
    gpuErrchk( cudaGetDeviceProperties(&cdp,0) );
    //int blockside = cdp.maxThreadsPerBlock >= 1024 ? 32 : 16; //length of the side of a block
    int blockside = 16;

    int thrno = cdp.maxThreadsPerBlock >= 1024 ? 512 : 256;

    int no_side = strno / blockside + (strno % blockside == 0 ? 0 : 1); // number of sides in a square grid
    int no_blocks = no_side + (no_side * no_side - no_side)/2; //number of blocks in a square grid
    int v3_noblocks = strno / thrno + (strno % thrno == 0 ? 0 : 1);
    
    //load
    unsigned int *arr = 0; //32* int32 == 1024 bits
    bool *rarr=0;

    unsigned int *dev_arr=0;
    bool *dev_rarr=0;

    arr = (unsigned int*)std::calloc(1, strlen * strno * sizeof(unsigned int ));
    //arr = (unsigned int*)std::malloc( strlen * strno * sizeof(unsigned int ));
    if (display)
    {
        rarr = (bool*)std::calloc(1, strno * strno * sizeof(bool));
    }
    
    //varr = (int*)std::malloc(strlen * strno * sizeof( int ));
    //std::bitset<STRLEN> bs;
    inputFile.open(path);
    std::getline(inputFile, inStr);
    int blockIdx = 0 ;
    for (size_t i = 0; i < strno; i++)
    {
        blockIdx = i/blockside;
        if (!inputFile.eof())
        {
            std::getline(inputFile, inStr);
            if (display)
            {
                strings[i] = inStr;
            }
            //std::cout<<inStr<<std::endl;
            for (size_t k = 0; k < strlen; k++)
            {
                //arr[i * arrlen + (k/intsize)]<<=1; //arrlen?
                //arr[i * arrlen + (k/intsize)] |= (inStr[k] - '0');
                //^ wczytuje ok

                //arr zmainia indeksacji
                arr[(k/intsize) * strno + i]<<=1;
                arr[(k/intsize) * strno + i] |= (inStr[k] - '0');
                /*if (k%intsize==31)
                {
                    std::cout<<std::bitset<32>(arr[(k/intsize) * strno + i])<<std::endl;
                }*/
            }
            
        }
    }
    inputFile.close();
    

    //malloc
    cudaStatus = cudaMalloc((void**)&dev_arr, strlen * strno * sizeof(int));
    gpuErrchk(cudaStatus);

    if (display)
    {
        cudaStatus = cudaMalloc((void**)&dev_rarr, strno * strno * sizeof(bool));
        gpuErrchk(cudaStatus);
    }
    

    cudaStatus = cudaMemcpy(dev_arr, arr, strlen * strno * sizeof(int), cudaMemcpyHostToDevice);
    gpuErrchk(cudaStatus);

    auto l2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> load_double = l2 - l1;
    std::cout <<"load in: "<< load_double.count() << " ms\n";


    //CUDA calculations
    cudaEventRecord(start);
    CompareWordsV3<<<v3_noblocks, thrno,48000 >>>(dev_arr, strno, arrlen, thrno, display, dev_rarr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"gpu processed in: "<<milliseconds<<" ms\n";

    cudaStatus = cudaMemcpyFromSymbol(&counter, dev_counter,sizeof(long long int),0,cudaMemcpyDeviceToHost);
    gpuErrchk(cudaStatus);


    if (!display)
    {
        std::cout<<counter<<std::endl;
    }
    else
    {
        cudaStatus = cudaMemcpy(rarr, dev_rarr, strno * strno * sizeof(bool), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaStatus);

        std::cout<<counter<<std::endl;
        for (size_t i = 0; i < strno; i++)
        {
            for (size_t j = i+1; j < strno; j++)
            {
                
                if (i*strno + j < strno*strno &&  rarr[i*strno + j])
                {
                    //std::cout<<i << " "<<j <<std::endl;
                    std::cout<<strings[i] << " "<<strings[j] <<std::endl;
                }
            }
        }
    }

    //local calculations
    if (local)
    {
        //clear arrays
        if (display)
        {
            for (size_t i = 0; i < strno; i++)
            {
                for (size_t j = i+1; j < strno; j++)
                {
                
                    if (i*strno + j < strno*strno)
                    {
                        rarr[i*strno + j] = false;
                    }
                }
            }
        }
        auto p1 = std::chrono::high_resolution_clock::now();
        counter = compareWordsLocal(arr,strno,arrlen,thrno, display, rarr);
        auto p2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> proc_double = p2 - p1;
        std::cout <<"locally processed in: "<< proc_double.count() << " ms\n";

       if (display)
       {
           std::cout<<counter<<std::endl;
            for (size_t i = 0; i < strno; i++)
            {
                for (size_t j = i+1; j < strno; j++)
                {
                
                    if (i*strno + j < strno*strno &&  rarr[i*strno + j])
                    {
                        //std::cout<<i << " "<<j <<std::endl;
                        std::cout<<strings[i] << " "<<strings[j] <<std::endl;
                    }
                }
            }
       }else{
            std::cout<<counter<<std::endl;
       }
    }
    

    cudaStatus = cudaGetLastError();
    gpuErrchk(cudaStatus);

    cudaFree(dev_arr);
    if (display)
    {
        cudaFree(dev_rarr);
        free(rarr);
    }
    
    free(arr);
    

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}