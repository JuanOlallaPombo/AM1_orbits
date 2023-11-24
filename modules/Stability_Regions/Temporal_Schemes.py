from scipy.optimize import newton

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
