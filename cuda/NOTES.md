# FrodoKEM-640 CUDA notes (Colab, T4, AES128 matrix A)

## Whole-KEM baselines (test_KEM, Xeon @ 2.2 GHz)
| Build | Keygen | Encaps | Decaps |
|---|---|---|---|
| AVX2 (fast) | ~511-567 us | ~592-595 us | ~556-561 us |
| Plain-C reference | 1307 us | 6252 us | 5877 us |

## Multiply kernel (A*s+e, N=640, NBAR=8), GPU kernel time only
| Batch | v1 | v2 | v3 | CPU plain-C loop |
|---|---|---|---|---|
| 1 | 0.293 ms | 0.022 ms | 0.011 ms | 0.193 ms |
| 16 | 2.89 ms | 0.194 ms | 0.118 ms | 3.27 ms |
| 256 | 32.2 ms | 2.76 ms | 1.74 ms | 51.6 ms |

v3 at batch 256: ~964 GOP/s (counting multiply+add as 2 ops). Host-to-device copy at batch 256: ~1.6 ms.

## Files
- matmul_test.cu: v1, one thread per output element (uncoalesced reads)
- matmul_test_v2.cu: block per row, coalesced reads, warp-shuffle reduction
- matmul_test_v3.cu: 4 rows of A per block, reuses s values

## Caveats
- CPU baseline here is a plain-C loop, not the repo's AVX2 code.
- A generation (AES) not yet measured or ported. Gen/mul split still to do.
- Not yet integrated into the repo or checked against the KATs.
- Colab CPU timings are noisy; quote medians.
