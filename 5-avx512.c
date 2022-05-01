#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#define UNROLL (4)
#define BLOCKSIZE 32

void do_block(int n, int si, int sj, int sk, double *A, double *B, double *C) {
    for (int i = si; i < si+BLOCKSIZE; i+=UNROLL * 8) {
        for (int j = sj; j < sj+BLOCKSIZE; ++j) {
            __m512d c[UNROLL];
            for (int r = 0; r < UNROLL; r++) {
                c[r] = _mm512_load_pd(C+i+r*8+j*n);
            }
            for (int k = sk; k < sk+BLOCKSIZE; k++) {
                __m512d bb = _mm512_broadcastsd_pd(_mm_load_sd(B+j*n+k));
                for (int r = 0; r < UNROLL; r++) {
                    c[r] = _mm512_fmadd_pd(_mm512_load_pd(A+n*k+r*8+i), bb, c[r]);
                }
            }
            for (int r = 0; r < UNROLL; r++) {
                _mm512_store_pd(C+i+r*8+j*n, c[r]);
            }
        }
    }
}

void dgemm(int n, double *A, double *B, double *C) {
    for (int sj = 0; sj < n; sj +=BLOCKSIZE)
        for (int si = 0; si < n; si +=BLOCKSIZE)
            for (int sk = 0; sk < n; sk +=BLOCKSIZE)
                do_block(n, si, sj, sk, A, B, C);
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    double* A = (double*)aligned_alloc(sizeof(__m512d), sizeof(double) * n * n);
    double* B = (double*)aligned_alloc(sizeof(__m512d), sizeof(double) * n * n);
    double* C = (double*)aligned_alloc(sizeof(__m512d), sizeof(double) * n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i+j*n] = 10 * i + j;
            B[i+j*n] = i + 10 * j;
            C[i+j*n] = 0;
        }
    }
    dgemm(n, A, B, C);
    printf("%f\n", C[n * n - 1]);
}
