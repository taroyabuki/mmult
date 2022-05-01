import numpy as np
import time

n = 4000
A = [[10.0 * r + c for c in range(n)] for r in range(n)]
B = [[r + 10.0 * c for c in range(n)] for r in range(n)]
C = [[0] * n for _ in range(n)]

start_time = time.time()
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i][j] += A[i][k] * B[k][j]
mid_time = time.time()
print(f'{n},{mid_time - start_time},{C[n-1][n-1]}')

c = np.matmul(A, B)
end_time = time.time()
print(f'{n},{end_time - mid_time},{c[n-1][n-1]}')
