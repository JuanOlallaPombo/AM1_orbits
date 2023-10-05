from numpy import zeros, array
import matplotlib.pyplot as plt
from Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler

#Cauchy Problem solving function. Takes a temporal scheme, initial conditions,
#a derivative function (F), a time domain and a timestep as inputs.
def F_Cauchy(Temporal_Scheme, U_0, F, tf, dt):
    N=int(tf/dt)
    U = zeros((len(U_0), N))
    U[:,0] = U_0
    if Temporal_Scheme==Euler:
        for t in range(0,N-1):
            U[:,t+1] = Euler(U[:,t],F,dt,t)
    elif Temporal_Scheme==Crank_Nicolson:
        for t in range(0,N-1):
            U[:,t+1] = Crank_Nicolson(U[:,t],F,dt,t)
    elif Temporal_Scheme==RK4:
        for t in range(0,N-1):
            U[:,t+1] = RK4(U[:,t],F,dt,t)
    elif Temporal_Scheme==Inverse_Euler:
        for t in range(0,N-1):
            U[:,t+1] = Inverse_Euler(U[:,t],F,dt,t)
    return U

#Derivative Function for Kepler's orbiting problem.
def F_Kepler(U, t):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

#Application to a real problem
tf = 10                 #Time interval, here marked as final time since time starts on 0
dt = 0.001              #Timestep
N = int(tf/dt)

U_0 = array([1,0,0,1])  #Setting initial conditions, starts in point (1,0) with vertical velocity
x = array(zeros(N))
y = array(zeros(N))

U = F_Cauchy(Crank_Nicolson, U_0, F_Kepler, tf, dt)
x[:] = U[0,:]
y[:] = U[1,:]

plt.figure(1)
plt.title('Trajectory')
plt.axis('equal')
plt.plot(x,y)
plt.show()
