from numpy import zeros, array
import matplotlib.pyplot as plt
from Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler
from Cauchy_Problem import F_Cauchy

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

#Solving Cauchy Problem
U = F_Cauchy(Crank_Nicolson, U_0, F_Kepler, tf, dt)
x[:] = U[0,:]
y[:] = U[1,:]

#Graph with trajectory results
plt.figure(1)
plt.title('Trajectory')
plt.axis('equal')
plt.plot(x,y)
plt.show()