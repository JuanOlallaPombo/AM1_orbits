import Temporal_Schemes as TS
from numpy import array, zeros
from Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler, Embedded_RK

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
    elif Temporal_Scheme==Embedded_RK:
        for t in range(0,N-1):
            U[:,t+1] = Embedded_RK(U[:,t],dt,t*dt,F,8,1e-6)        #Order 3 RK method
    return U

