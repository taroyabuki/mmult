#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

void dgemm(int n, double *A, double *B, double *C) {
    for (int i = 0; i < n; i+=8) {
        for (int j = 0; j < n; ++j) {
            __m512d c0 = _mm512_load_pd(C+i+j*n);
            for (int k = 0; k < n; k++) {
                __m512d bb = _mm512_broadcastsd_pd(_mm_load_sd(B+j*n+k));
                c0 = _mm512_fmadd_pd(_mm512_load_pd(A+n*k+i), bb, c0);
            }
            _mm512_store_pd(C+i+j*n, c0);
        }
    }
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
