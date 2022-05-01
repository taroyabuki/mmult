#include <stdio.h>
#include <stdlib.h>

void dgemm(int n, double *A, double *B, double *C) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double cij = C[i+j*n];
            for (int k = 0; k < n; k++) {
                cij += A[i+k*n] * B[k+j*n];
            }
            C[i+j*n] = cij;
        }
    }
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    double* A = (double*)malloc(sizeof(double) * n * n);
    double* B = (double*)malloc(sizeof(double) * n * n);
    double* C = (double*)malloc(sizeof(double) * n * n);
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
