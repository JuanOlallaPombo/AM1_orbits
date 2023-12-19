from numpy import array
import math

## ODEs
# Sistema caótico con k<1 
def ODE_Duffing(U, t, a, b, c): 
    x, y = U[0], U[1]

    vx=y
    vy=-x**3-0.01*y+math.cos(t)


    return array( [vx, vy] )

def ODE_Van_Der_Poel(U, t, a, b, c): 
    x, y = U[0], U[1]

    vx=y
    vy=-a*(x**2-1)*y-x

    return array( [vx, vy] )

def ODE_Rayleigh(U, t, a, b, c): 
    x, y = U[0], U[1]

    vx=y
    vy=x-x**3-y+a*math.cos(t)

    return array( [vx, vy] )

def Lorentz(U, t, a, b, c): 
    x, y, z= U[0], U[1], U[2]
   
    vx= -c*x+c*y
    vy = b*x -x*z-y
    vz = -b*z + x*y


    return array( [vx, vy, vz] )

def Rossler(U, t, a, b, c): 
    x, y, z= U[0], U[1], U[2]
    
    vx = -y- z
    vy = a*y + x
    vz = b + z*(x-c)

    return array( [vx, vy, vz] )
