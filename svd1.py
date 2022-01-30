import numpy as np
import numpy.linalg as la
def calculate(A):
    m = A.shape[0]
    n = A.shape[1]
    S = np.zeros(n)

    # finding eigenvectors with biggest eigenvalues of A*transpose(A)
    helper = np.dot(A.T, A)
    eigenvalues, eigenvectors = la.eigh(helper)
    # descending sort of all the eigenvectors according to their eigenvalues
    index = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    V = eigenvectors


    # S is a diagonal matrix that keeps square root of eigenvalues
    j = 0
    for i in eigenvalues:
        if j == S.size:
            break
        elif i >= 0:
            S[j] = np.sqrt(i)
            j += 1


    sigma = np.zeros((m, n))
    for i in range(m):
        sigma[i][i] = np.sqrt(eigenvalues[i])

    U = np.empty((m, m), float)
    for i in range (m):
        U[i] = np.dot(A, V[i])/S[i]


    return U, sigma, V.T
