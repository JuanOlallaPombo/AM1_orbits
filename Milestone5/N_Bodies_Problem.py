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

    drdt[:] = v[:]                                #Derivative of position equals velocity

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

U_0 = array([1,0,0,1,0,0,0,-1,1,0,0,1,0,1,0,0,1,-1])  #Setting initial conditions
#In order to understand what each coordinate means in terms of positions and velocities
#check variable separation starting in code line 58
x1 = array(zeros((N)))
y1 = array(zeros((N)))
z1 = array(zeros((N)))
x2 = array(zeros((N)))
y2 = array(zeros((N)))
z2 = array(zeros((N)))
x3 = array(zeros((N)))
y3 = array(zeros((N)))
z3 = array(zeros((N)))

#Cauchy problem solving
U = F_Cauchy(Crank_Nicolson, U_0, F_Nbodies, tf, dt)

x1[:] = U[0, :]
x2[:] = U[2, :]
x3[:] = U[4, :]
y1[:] = U[6, :]
y2[:] = U[8, :]
y3[:] = U[10, :]
z1[:] = U[12, :]    
z2[:] = U[14, :]
z3[:] = U[16, :]

#Showing results
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_title('N bodies problem')
ax1.plot(x1,y1,z1,'g')
ax1.plot(x2,y2,z2,'r')
ax1.plot(x3,y3,z3,'b')
ax1.set_aspect('auto')
plt.show()