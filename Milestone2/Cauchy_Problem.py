from numpy import zeros

#Cauchy Problem solving function. Takes a temporal scheme, initial conditions,
#a derivative function (F), a time domain and a timestep as inputs.
def F_Cauchy(Temporal_Scheme, U_0, F, tf, dt):
    N=int(tf/dt)
    U = zeros((len(U_0), N))
    U[:,0] = U_0
    for t in range(0,N-1):
       U[:,t+1] = Temporal_Scheme(U[:,t],F,dt,t)
    return U

