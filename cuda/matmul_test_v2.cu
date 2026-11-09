#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define N 640
#define NBAR 8

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

// out[b][i][k] = E[b][i][k] + sum_j A[i][j] * S[b][k][j]   (wraps mod 2^16, like the repo)
void cpu_mul(uint16_t *out, const uint16_t *A, const uint16_t *S, const uint16_t *E, int batch) {
  for (int b = 0; b < batch; b++)
    for (int i = 0; i < N; i++)
      for (int k = 0; k < NBAR; k++) {
        uint16_t sum = 0;
        for (int j = 0; j < N; j++)
          sum += A[i*N + j] * S[(size_t)b*NBAR*N + k*N + j];
        size_t o = (size_t)b*N*NBAR + i*NBAR + k;
        out[o] = E[o] + sum;
      }
}

// v1: one thread per output element
__global__ void gpu_mul(uint16_t *out, const uint16_t *A, const uint16_t *S, const uint16_t *E) {
  int k = threadIdx.x;
  int i = blockIdx.x;
  int b = blockIdx.y;
  uint16_t sum = 0;
  const uint16_t *s = S + (size_t)b*NBAR*N + k*N;
  for (int j = 0; j < N; j++) sum += A[i*N + j] * s[j];
  size_t o = (size_t)b*N*NBAR + i*NBAR + k;
  out[o] = E[o] + sum;
}


__global__ void gpu_mul_v2(uint16_t *out, const uint16_t *A, const uint16_t *S, const uint16_t *E) {
  int i = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;   // 128 threads
  const uint16_t *a = A + (size_t)i*N;
  const uint16_t *s = S + (size_t)b*NBAR*N;
  unsigned int acc[NBAR];
  #pragma unroll
  for (int k = 0; k < NBAR; k++) acc[k] = 0;
  for (int j = tid; j < N; j += 128) {
    unsigned int av = a[j];
    #pragma unroll
    for (int k = 0; k < NBAR; k++) acc[k] += av * (unsigned int)s[k*N + j];
  }
  #pragma unroll
  for (int k = 0; k < NBAR; k++)
    for (int off = 16; off > 0; off >>= 1)
      acc[k] += __shfl_down_sync(0xffffffff, acc[k], off);
  __shared__ unsigned int part[4][NBAR];
  int lane = tid & 31, w = tid >> 5;
  if (lane == 0) {
    #pragma unroll
    for (int k = 0; k < NBAR; k++) part[w][k] = acc[k];
  }
  __syncthreads();
  if (tid < NBAR) {
    unsigned int t = part[0][tid] + part[1][tid] + part[2][tid] + part[3][tid];
    size_t o = (size_t)b*N*NBAR + (size_t)i*NBAR + tid;
    out[o] = (uint16_t)(E[o] + t);
  }
}

static double now_ms() {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec*1e3 + t.tv_nsec/1e6;
}

int main(int argc, char **argv) {
  int batch = argc > 1 ? atoi(argv[1]) : 1;
  size_t szA = (size_t)N*N, szS = (size_t)batch*NBAR*N, szO = (size_t)batch*N*NBAR;
  uint16_t *A = (uint16_t*)malloc(szA*2), *S = (uint16_t*)malloc(szS*2), *E = (uint16_t*)malloc(szO*2);
  uint16_t *oc = (uint16_t*)malloc(szO*2), *og = (uint16_t*)malloc(szO*2);
  srand(1);
  for (size_t i = 0; i < szA; i++) A[i] = rand() & 0xFFFF;
  for (size_t i = 0; i < szS; i++) S[i] = rand() & 0xFFFF;
  for (size_t i = 0; i < szO; i++) E[i] = rand() & 0xFFFF;

  // CPU timing: median of 5 runs
  double t[5];
  for (int r = 0; r < 5; r++) { double a = now_ms(); cpu_mul(oc, A, S, E, batch); t[r] = now_ms() - a; }
  for (int x = 0; x < 5; x++) for (int y = x+1; y < 5; y++) if (t[y] < t[x]) { double q=t[x]; t[x]=t[y]; t[y]=q; }
  double cpu_ms = t[2];

  // GPU
  cudaFree(0);
  uint16_t *dA, *dS, *dE, *dO;
  CK(cudaMalloc(&dA, szA*2)); CK(cudaMalloc(&dS, szS*2)); CK(cudaMalloc(&dE, szO*2)); CK(cudaMalloc(&dO, szO*2));
  cudaEvent_t e0, e1, e2, e3;
  cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);

  cudaEventRecord(e0);
  CK(cudaMemcpy(dA, A, szA*2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dS, S, szS*2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dE, E, szO*2, cudaMemcpyHostToDevice));
  cudaEventRecord(e1);
  dim3 grid(N, batch), block(NBAR);
  gpu_mul_v2<<<grid, 128>>>(dO, dA, dS, dE);        // warm-up
  CK(cudaGetLastError());
  cudaEventRecord(e2);
  for (int r = 0; r < 20; r++) gpu_mul_v2<<<grid, 128>>>(dO, dA, dS, dE);
  cudaEventRecord(e3);
  CK(cudaEventSynchronize(e3));
  CK(cudaMemcpy(og, dO, szO*2, cudaMemcpyDeviceToHost));

  float h2d, kern;
  cudaEventElapsedTime(&h2d, e0, e1);
  cudaEventElapsedTime(&kern, e2, e3);
  kern /= 20;

  int ok = memcmp(oc, og, szO*2) == 0;
  double ops = 2.0 * N * N * NBAR * batch;
  double bytes = (double)(szA + szS + 2*szO) * 2;   // A + S + E read, out written
  printf("batch=%d  correct=%s\n", batch, ok ? "YES" : "NO");
  printf("CPU (plain C, -O2): %.3f ms  (%.3f GOP/s)\n", cpu_ms, ops/(cpu_ms*1e6));
  printf("GPU kernel:         %.4f ms  (%.3f GOP/s)\n", kern, ops/(kern*1e6));
  printf("GPU H2D copy:       %.3f ms\n", h2d);
  printf("Ideal arithmetic intensity: %.2f ops/byte\n", ops / bytes);
  return 0;
}
