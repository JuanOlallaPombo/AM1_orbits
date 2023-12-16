from numpy import array, zeros, float64
from numpy.linalg import eig

#Adding small perturbations to the system matrix as to test its stability
def System_matrix(F, U0, t):

    eps=1e-6
    N=len(U0)
    F_aux = F(U0,t)
    A = zeros((len(F_aux),len(F_aux)), dtype = float64)
    delta = zeros(N)

    for j in range(N):
        delta[:] = 0
        delta[j] = eps
        A[:,j] = (F(U0 + delta, t)-F(U0 - delta, t))/(2*eps)

    return A

#Computes the eigenvalues of the system matrix
def Eigenvalues_Jacobian(F, U0, t0):       
                     
    N = len(U0)
    A = zeros((N,N))
          
    A = System_matrix(F, U0, t0) 
    lambdaA, vectors =eig (A)

    return lambdaA

def test_System_matrix():
    U0 = [0., 0.]
