from scipy.optimize import newton
from numpy import array, zeros, float64, dot, linalg

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

def Embedded_RK( U, dt, t, F, q, Tolerance): 
  
    #(a, b, bs, c) = Butcher_array(q)
    #a, b, bs, c = Butcher_array(q)
 
    N_stages = { 2:2, 3:4, 8:13  }
    Ns = N_stages[q]
    a = zeros( (Ns, Ns), dtype=float64) 
    b = zeros(Ns); bs = zeros(Ns); c = zeros(Ns) 
   
    if Ns==2: 
     
     a[0,:] = [ 0, 0 ]
     a[1,:] = [ 1, 0 ] 
     b[:]  = [ 1/2, 1/2 ] 
     bs[:] = [ 1, 0 ] 
     c[:]  = [ 0, 1]  

    elif Ns==13: 
       c[:] = [ 0., 2./27, 1./9, 1./6, 5./12, 1./2, 5./6, 1./6, 2./3 , 1./3,   1., 0., 1.]

       a[0,:]  = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
       a[1,:]  = [ 2./27, 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0] 
       a[2,:]  = [ 1./36 , 1./12, 0., 0., 0., 0., 0.,  0.,0., 0., 0., 0., 0] 
       a[3,:]  = [ 1./24 , 0., 1./8 , 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
       a[4,:]  = [ 5./12, 0., -25./16, 25./16., 0., 0., 0., 0., 0., 0., 0., 0., 0]
       a[5,:]  = [ 1./20, 0., 0., 1./4, 1./5, 0., 0.,0., 0., 0., 0., 0., 0] 
       a[6,:]  = [-25./108, 0., 0., 125./108, -65./27, 125./54, 0., 0., 0., 0., 0., 0., 0] 
       a[7,:]  = [ 31./300, 0., 0., 0., 61./225, -2./9, 13./900, 0., 0., 0., 0., 0., 0] 
       a[8,:]  = [ 2., 0., 0., -53./6, 704./45, -107./9, 67./90, 3., 0., 0., 0., 0., 0] 
       a[9,:]  = [-91./108, 0., 0., 23./108, -976./135, 311./54, -19./60, 17./6, -1./12, 0., 0., 0., 0] 
       a[10,:] = [ 2383./4100, 0., 0., -341./164, 4496./1025, -301./82, 2133./4100, 45./82, 45./164, 18./41, 0., 0., 0] 
       a[11,:] = [ 3./205, 0., 0., 0., 0., -6./41, -3./205, -3./41, 3./41, 6./41, 0., 0., 0]
       a[12,:] = [ -1777./4100, 0., 0., -341./164, 4496./1025, -289./82, 2193./4100, 51./82, 33./164, 19./41, 0.,  1., 0]
      
       b[:]  = [ 41./840, 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 41./840, 0., 0.] 
       bs[:] = [ 0., 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 0., 41./840, 41./840]     
     

    
    k = RK_stages( F, U, t, dt, a, c )
    Error = dot( b-bs, k )

    dt_min = min( dt, dt * ( Tolerance / linalg.norm(Error) ) **(1/q) )
    N = int( dt/dt_min  ) + 1
    h = dt / N
    Uh = U.copy()

    for i in range(0, N): 

        k = RK_stages( F, Uh, t + h*i, h, a, c ) 
        Uh += h * dot( b, k )

    return Uh

def RK_stages( F, U, t, dt, a, c ): 

     k = zeros( (len(c), len(U)), dtype=float64 )

     for i in range(len(c)): 

        for  j in range(len(c)-1): 
          Up = U + dt * dot( a[i, :], k)

        k[i, :] = F( Up, t + c[i] * dt ) 

     return k 
