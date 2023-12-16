from numpy import array, zeros, float64

#Adding perturbations to the matrix as to test its stability
def System_matrix(F, U0, t):

    eps=1e-6
    N=len(U0)
    A = zeros((N,N), dtype=float64)
    delta = zeros(N)

    for j in range(N):
        delta[:] = 0
        delta[j] = eps
        A[:,j] = (F(U0 + delta, t)-F(U0 - delta, t))/(2*eps)

    return A

def test_System_matrix():
    U0 = [0., 0.]
