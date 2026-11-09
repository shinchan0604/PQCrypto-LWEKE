#include <stdint.h>
#include <cuda_runtime.h>
#define N 640
#define NBAR 8

#define R 4
__global__ void gpu_mul_v3(uint16_t *out, const uint16_t *A, const uint16_t *S, const uint16_t *E) {
  int i0 = blockIdx.x * R, b = blockIdx.y, tid = threadIdx.x;   // 128 threads
  const uint16_t *s = S + (size_t)b*NBAR*N;
  unsigned int acc[R][NBAR];
  #pragma unroll
  for (int r = 0; r < R; r++)
    #pragma unroll
    for (int k = 0; k < NBAR; k++) acc[r][k] = 0;
  for (int j = tid; j < N; j += 128) {
    unsigned int sv[NBAR];
    #pragma unroll
    for (int k = 0; k < NBAR; k++) sv[k] = s[k*N + j];
    #pragma unroll
    for (int r = 0; r < R; r++) {
      unsigned int av = A[(size_t)(i0 + r)*N + j];
      #pragma unroll
      for (int k = 0; k < NBAR; k++) acc[r][k] += av * sv[k];
    }
  }
  #pragma unroll
  for (int r = 0; r < R; r++)
    #pragma unroll
    for (int k = 0; k < NBAR; k++)
      for (int off = 16; off > 0; off >>= 1)
        acc[r][k] += __shfl_down_sync(0xffffffff, acc[r][k], off);
  __shared__ unsigned int part[4][R][NBAR];
  int lane = tid & 31, w = tid >> 5;
  if (lane == 0) {
    #pragma unroll
    for (int r = 0; r < R; r++)
      #pragma unroll
      for (int k = 0; k < NBAR; k++) part[w][r][k] = acc[r][k];
  }
  __syncthreads();
  if (tid < R*NBAR) {
    int r = tid / NBAR, k = tid % NBAR;
    unsigned int t = part[0][r][k] + part[1][r][k] + part[2][r][k] + part[3][r][k];
    size_t o = (size_t)b*N*NBAR + (size_t)(i0 + r)*NBAR + k;
    out[o] = (uint16_t)(E[o] + t);
  }
}


extern "C" int frodo_cuda_mul_add_as_plus_e(uint16_t *out, const uint16_t *A, const uint16_t *s, const uint16_t *e) {
  static uint16_t *dA = 0, *dS, *dE, *dO;
  if (!dA) {
    if (cudaMalloc(&dA, N*N*2) || cudaMalloc(&dS, NBAR*N*2) || cudaMalloc(&dE, N*NBAR*2) || cudaMalloc(&dO, N*NBAR*2)) return 0;
  }
  cudaMemcpy(dA, A, N*N*2, cudaMemcpyHostToDevice);
  cudaMemcpy(dS, s, NBAR*N*2, cudaMemcpyHostToDevice);
  cudaMemcpy(dE, e, N*NBAR*2, cudaMemcpyHostToDevice);
  gpu_mul_v3<<<dim3(N/R, 1), 128>>>(dO, dA, dS, dE);
  if (cudaGetLastError() != cudaSuccess) return 0;
  cudaMemcpy(out, dO, N*NBAR*2, cudaMemcpyDeviceToHost);
  return 1;
}
