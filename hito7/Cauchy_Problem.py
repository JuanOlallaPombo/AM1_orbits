# Módulo para resolver el problema de Cauchy
from Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler
from numpy import zeros

# Cauchy Problem solving function. Takes a temporal scheme, initial conditions,
# a derivative function (F), a time domain and a timestep as inputs.
def F_Cauchy(Temporal_Scheme, U_0, F, tf, dt, a=None, b =None, c = None):

    print(Temporal_Scheme)
    print(F)
    N = int(tf / dt)
    
    # Verificar que U_0 tiene elementos en el eje 1
    if len(U_0) == 0:
        print("Error: U_0 está vacío.")
        return None
    
    U = zeros((len(U_0), N))
    
    # Verificar que U tiene al menos una columna
    if U.shape[1] > 0:
        U[:, 0] = U_0
    else:
        print("Error: La matriz U no tiene columnas.")
        return None
    
    if Temporal_Scheme == "Euler":
        for t in range(0, N - 1):
            U[:, t + 1] = Euler(U[:, t], F, dt, t, a, b , c)
    elif Temporal_Scheme == "Crank_Nicolson":
        for t in range(0, N - 1):
            U[:, t + 1] = Crank_Nicolson(U[:, t], F, dt, t, a, b , c )
    elif Temporal_Scheme == "RK4":
        for t in range(0, N - 1):
            U[:, t + 1] = RK4(U[:, t], F, dt, t, a, b , c )
    elif Temporal_Scheme == "Inverse_Euler":
        for t in range(0, N - 1):
            U[:, t + 1] = Inverse_Euler(U[:, t], F, dt, t, a, b , c )
    
    return U
