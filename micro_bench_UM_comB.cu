/* 
Author: Janki Bhimani
Northeastern University
Email: bhimanijanki@gmail.com
 */


#include <stdio.h>
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void kernel(float *a, int offset, int x)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int p;
  int t = offset + ((i*(x))+j);
{
  float q = (float)t;
  float s = sinf(q); 
  float c = cosf(q);
  a[t] = a[t] + sqrtf(s*s+c*c); //adding 1 to a
  for(p=0;p<1;p++)
  {
	q = sinf(q); 
  	q = cosf(q);
	q = sqrtf(s*s+c*c);
  }
}
}

float maxError(float *a, int n) 
{
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv)
{
  const int blockSize = 1024, nStreams = sqrt(atoi(argv[2]));
  int x = atoi(argv[1]);
  const int n = x *x * blockSize ;
  const int streamSize = n / nStreams/ nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);
   
  int devId = 0;
  if (argc > 3) devId = atoi(argv[3]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );
  dim3 block(32, 32);
  dim3 grid((sqrt(n))/32,(sqrt(n))/32); 
  dim3 grid1((sqrt(n))/nStreams/32, (sqrt(n))/nStreams/32);
  x= x* nStreams;

  float ms, msk, seq, aloc; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startaloc, stopaloc, startEvent, stopEvent, startKernel, stopKernel, dummyEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startaloc) );
  checkCuda( cudaEventCreate(&stopaloc) );
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&startKernel) );
  checkCuda( cudaEventCreate(&stopKernel) );
  checkCuda( cudaEventCreate(&dummyEvent) );
for (int i = 0; i < nStreams* nStreams; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );
   checkCuda( cudaEventRecord(startaloc,0) ); 
  float *a = (float*)malloc(bytes) ;     
  checkCuda( cudaMallocManaged((void**)&a, bytes) ); // device
  checkCuda( cudaEventRecord(stopaloc, 0) );
  checkCuda( cudaEventSynchronize(stopaloc) );
  checkCuda( cudaEventElapsedTime(&aloc, startaloc, stopaloc) );
  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  checkCuda( cudaEventRecord(startKernel,0) );
  kernel<<<grid,block>>>(a, 0, sqrt(n));
    checkCuda(cudaDeviceSynchronize());
  checkCuda( cudaEventRecord(stopKernel, 0) );
  checkCuda( cudaEventSynchronize(stopKernel) );
  checkCuda( cudaEventElapsedTime(&msk, startKernel, stopKernel) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
seq = ms;  
printf("Time for seq transfer and execute (ms): %f\n", ms+aloc);
printf("Time for kernel execute (ms): %f\n", msk);
printf("Bytes for sequential transfer (bytes): %d\n", bytes);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams* nStreams; ++i) {
    int offset = i * streamSize;
    kernel<<<grid1, block, 0, stream[i]>>>(a, offset, sqrt(n)/nStreams);
    checkCuda(cudaDeviceSynchronize());
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Scheduling scheme type I transfer and execute (ms): %f\n", ms+aloc);
  printf("  max error: %e\n", maxError(a, n));

  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );

  for (int i = 0; i < nStreams* nStreams; ++i)
  {
    int offset = i * streamSize;
    kernel<<<grid1, block, 0, stream[i]>>>(a, offset, sqrt(n)/nStreams);
    checkCuda(cudaDeviceSynchronize());
  }
 
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Scheduling scheme type II transfer and execute (ms): %f\n", ms+aloc);
  printf("  max error: %e\n", maxError(a, n));
  printf("% Overlap (%): %f\n", (seq-ms)/seq*100);
  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaEventDestroy(dummyEvent) );
  for (int i = 0; i < nStreams* nStreams; ++i)
    checkCuda( cudaStreamDestroy(stream[i]) );
  cudaFree(a);
  //cudaFree(a);

  return 0;
}

