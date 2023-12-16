from numpy import array, zeros, reshape
from numpy.linalg import norm
from Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler, Embedded_RK
from Cauchy_Problem import F_Cauchy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import newton
from Stability import Eigenvalues_Jacobian

#Specify the number of bodies
Nb = 3

#Specify the number of coordinates
Nc = 3

#N-body Problem Equations
def F_Nbodies(U1, t):
    UT = [1,0,0,0,0,0]
    US = [0,0,0,0,0,0]
    U = zeros((2*Nb*Nc))
    Us = reshape(U, (Nb,Nc,2))              #Creating first pointer
    r = reshape(Us[:,:,0], (Nb,Nc))         #Pointer for position
    r [0,:] = U1[0:3]
    r [1,:] = UT[0:3]
    r [2,:] = US[0:3]
    v = reshape(Us[:,:,1], (Nb,Nc))         #Pointer for velocity
    v [0,:] = U1[3:6]
    v [1,:] = UT[3:6]
    v [2,:] = US[3:6]

    F = zeros((2*Nb*Nc))                    #Derivative matrix
    Fs = reshape(F, (Nb,Nc,2))              #Creating pointer for F
    drdt = reshape(Fs[:,:,0], (Nb,Nc))      #Pointer for position derivative
    dvdt = reshape(Fs[:,:,1], (Nb,Nc))      #Pointer for velocity derivative

    drdt[:,:] = v[:,:]                                #Derivative of position equals velocity

    for i in range(0,Nb): 
        for j in range(0,Nb):
            if i==j:
                break                       #Bodies do not exert forces upon themselves
            else:
                denom = ((r[j,0]-r[i,0])**2+(r[j,1]-r[i,1])**2+(r[j,2]-r[i,2])**2)**0.5
                for k in range(0,Nc):
                    dvdt[i,k] = (r[j,k] - r[i,k])/denom
    return array([drdt[0,0],drdt[0,1],drdt[0,2],dvdt[0,0],dvdt[0,1],dvdt[0,2]])

#Cauchy Problem for 3 bodies (Example)
tf = 10
dt = 0.001
N = int(tf/dt)

U0 = zeros((5,6))
U0[0,:] = [ 0.8, 0.6, 0., 0., 0., 0.  ]
U0[1,:] = [ 0.8, -0.6, 0., 0., 0., 0.  ]
U0[2,:] = [ -0.1, 0.0, 0., 0., 0., 0.  ]
U0[3,:] = [ 0.1, 0.0, 0., 0., 0., 0.  ]
U0[4,:] = [ 1.1, 0.0, 0., 0., 0., 0.  ]

#U_solve = F_Cauchy(Crank_Nicolson, U_0, F_Nbodies, tf, dt)

#Lagrange Points
for i in range(0,5):
    Lagrange = array(zeros(len(U0[i,:])))
    def F_Nbodies_RES(U0):
        U1 = zeros((6))
        U1[0:3] = U0
        U1[3:6] = [0,0,0]
        F = F_Nbodies(U1,0)
        return F[3:6]
    Lagrange = newton(F_Nbodies_RES,U0[i,0:3])
    print(Lagrange)

    #Stability analysis based on eigenvalues of A
    #lambdaAs = zeros((3))
    #lambdaAs = Eigenvalues_Jacobian(F_Nbodies, U0[i,0:3],0)
    #for lambdaA in lambdaAs:
    #    print('Eigenvalue of A = ', lambdaA)

    eps = array([1e-6,1e-6,1e-6])


#Orbits around Lagrange Points with an Embedded Runge-Kutta method
    U_0 = zeros((6))
    U_0[0:3] = Lagrange + eps
    U_0[3:6] = eps
    U_L = F_Cauchy(Embedded_RK,U_0, F_Nbodies, tf, dt)

    x1 = array(zeros((N)))
    y1 = array(zeros((N)))
    z1 = array(zeros((N)))

    x1[:] = U_L[0,:]
    y1[:] = U_L[1,:]
    z1[:] = U_L[2,:]

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot(x1,y1,z1,'g')
    ax1.set_aspect('auto')
    plt.show()
