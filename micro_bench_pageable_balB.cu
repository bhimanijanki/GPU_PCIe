/* 
Author: Janki Bhimani
Northeastern University
Email: bhimanijanki@gmail.com
 */

#include <stdio.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
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
  //int k = offset + blockIdx.z*blockDim.z + threadIdx.z;
  int p;
  int t = offset + ((i*(x))+j);//((k*x*x)+(j*(x))+i);
{
  float q = (float)t;
  float s = sinf(q); 
  float c = cosf(q);
  a[t] = a[t] + sqrtf(s*s+c*c); //adding 1 to a
  for(p=0;p<28;p++)
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
 const int blockSize = 1024, nStreams = sqrt(2);
  int x = atoi(argv[1]);
  const int n = x *x * blockSize * nStreams* nStreams;
  const int streamSize = n / nStreams/ nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);
   
   
  int devId = 0;
  if (argc > 2) devId = atoi(argv[2]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );
  dim3 block(32, 32);
  dim3 grid((sqrt(n))/32,(sqrt(n))/32); 
  dim3 grid1((sqrt(n))/nStreams/32, (sqrt(n))/nStreams/32);
  x= x* nStreams;
  // allocate pinned host memory and device memory
  float *d_a;
  float *a = (float*)malloc(bytes) ;     
  checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device

  float ms, msk, seq; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startEvent, stopEvent, startKernel, stopKernel, dummyEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&startKernel) );
  checkCuda( cudaEventCreate(&stopKernel) );
  checkCuda( cudaEventCreate(&dummyEvent) );
for (int i = 0; i < nStreams* nStreams; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );
  
  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  checkCuda( cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaEventRecord(startKernel,0) );
  kernel<<<grid,block>>>(d_a, 0, sqrt(n));
  checkCuda( cudaEventRecord(stopKernel, 0) );
  checkCuda( cudaEventSynchronize(stopKernel) );
  checkCuda( cudaEventElapsedTime(&msk, startKernel, stopKernel) );
  checkCuda( cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
seq = ms;  
printf("Time for seq transfer and execute (ms): %f\n", ms);
printf("Time for kernel execute (ms): %f\n", msk);
printf("Bytes for sequential transfer (bytes): %d\n", bytes);
  printf("  max error: %e\n", maxError(a, n));

  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams* nStreams; ++i) {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice, 
                               stream[i]) );
    kernel<<<grid1, block, 0, stream[i]>>>(d_a, offset, sqrt(n)/nStreams);
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Scheduling scheme type I transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams* nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
  }
  for (int i = 0; i < nStreams* nStreams; ++i)
  {
    int offset = i * streamSize;
    kernel<<<grid1, block, 0, stream[i]>>>(d_a, offset, sqrt(n)/nStreams);
  }
  for (int i = 0; i < nStreams* nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Scheduling scheme type II transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));
  printf("% Overlap (%): %f\n", (seq-ms)/seq*100);
  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaEventDestroy(dummyEvent) );
  for (int i = 0; i < nStreams* nStreams; ++i)
    checkCuda( cudaStreamDestroy(stream[i]) );
  cudaFree(d_a);
  cudaFree(a);

  return 0;
}

