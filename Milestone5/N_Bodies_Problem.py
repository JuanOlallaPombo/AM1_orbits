from numpy import array, zeros, reshape
from numpy.linalg import norm
from Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler
from Cauchy_Problem import F_Cauchy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Specify the number of bodies
Nb = 3

#Specify the number of coordinates
Nc = 3

#N-body Problem Equations
def F_Nbodies(U, t):
    Us = reshape(U, (Nb,Nc,2))              #Creating first pointer
    r = reshape(Us[:,:,0], (Nb,Nc))         #Pointer for position 
    v = reshape(Us[:,:,1], (Nb,Nc))         #Pointer for velocity

    F = zeros((2*Nb*Nc))                    #Derivative matrix
    Fs = reshape(F, (Nb,Nc,2))              #Creating pointer for F
    drdt = reshape(Fs[:,:,0], (Nb,Nc))      #Pointer for position derivative
    dvdt = reshape(Fs[:,:,1], (Nb,Nc))      #Pointer for velocity derivative

    drdt = v                                #Derivative of position equals velocity

    for i in range(0,Nb): 
        for j in range(0,Nb):
            if i==j:
                break                       #Bodies do not exert forces upon themselves
            else:
                denom = ((r[j,0]-r[i,0])**2+(r[j,1]-r[i,1])**2+(r[j,2]-r[i,2])**2)**0.5
                for k in range(0,Nc):
                    dvdt[i,k] = (r[j,k] - r[i,k])/denom
    return F

#Cauchy Problem for 3 bodies (Example)
tf = 10
dt = 0.001
N = int(tf/dt)

U_0 = array([[1,0,0],[0,1,0],[0,1,0],[-1,1,0],[0,0,1],[1,0,-1]])  #Setting initial conditions, starts in point (1,0) with vertical velocity
r1 = array(zeros((Nc,N)))
r2 = array(zeros((Nc,N)))
r3 = array(zeros((Nc,N)))


U = F_Cauchy(Crank_Nicolson, U_0, F_Nbodies, tf, dt)
for i in range(0,N):
    r1[:,i] = U[0:Nc, i]
    r2[:,i] = U[Nc:2*Nc, i]
    r3[:,i] = U[2*Nc:3*Nc, i]

x1 = r1[0,:]
y1 = r1[1,:]
z1 = r1[2,:]

x2 = r2[0,:]
y2 = r2[1,:]
z2 = r2[2,:]

x3 = r3[0,:]
y3 = r3[1,:]
z3 = r3[2,:]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Trajectories')
for i in range(0,N):
    ax.scatter(x1[i], y1[i], z1[i])
    ax.scatter(x2[i], y2[i], z2[i])
    ax.scatter(x3[i], y3[i], z3[i])
plt.show()




