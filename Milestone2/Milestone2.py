from numpy import zeros, array
import matplotlib.pyplot as plt
from scipy.optimize import newton

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

#Temporal Schemes
def Euler(U, F, dt, t):
    U = U + dt * F(U, t*dt)
    return U

def Crank_Nicolson(U, F, dt, t):
    def CN_Res(x):
        return x - U_temp - dt/2 * F(x,(t+1)*dt)
    U_temp = U + dt/2 * F(U,t*dt)
    U = newton(CN_Res, U)
    return U

def RK4(U, F, dt, t):
    k1 = F(U,t*dt)
    k2 = F(U+0.5*dt*k1, dt*(t+1/2))
    k3 = F(U+0.5*dt*k2, dt*(t+1/2))
    k4 = F(U+dt*k3, (t+1)*dt)
    U = U + dt/6*(k1+2*k2+2*k3+k4)
    return U

def Inverse_Euler(U, F, dt, t):
    def IE_Res(x):
        return x - dt*F(x,(t+1)*dt) - U_temp
    U_temp = U
    U = newton(IE_Res, U)
    return U

#Application to a real problem
tf = 10                 #Time interval, here marked as final time since time starts on 0
dt = 0.001              #Timestep
N = int(tf/dt)

U_0 = array([1,0,0,1])  #Setting initial conditions, starts in point (1,0) with vertical velocity
x = array(zeros(N))
y = array(zeros(N))

U = F_Cauchy(Inverse_Euler, U_0, F_Kepler, tf, dt)
x[:] = U[0,:]
y[:] = U[1,:]

plt.figure(1)
plt.title('Trajectory')
plt.axis('equal')
plt.plot(x,y)
plt.show()
