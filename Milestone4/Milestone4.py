from Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler
from Stability_Regions import Stability_Region
from Cauchy_Problem import F_Cauchy
from numpy import array, zeros, linspace, abs, transpose, float64
import matplotlib.pyplot as plt

def F(U, t):
    x, y = U[0], U[1]
    return array([y, -x])

t=100
dt=0.001
N=int(t/dt)

U_0=array([0.1,0])
x=array(zeros(N))
U= F_Cauchy(Inverse_Euler, U_0, F, t, dt)
x[:] = U[0,:]

Tiempo = [dt*i for i in range(0,N)]

plt.figure(1)
plt.title('Trajectory')
plt.plot(Tiempo,x)
plt.show()

rho, x, y = Stability_Region(Inverse_Euler, 100, -4, 2, -4, 4)
plt.contour(x, y, transpose(rho), linspace(0,1,11))
plt.plot(0,dt,'r+')
plt.plot(0,-dt,'r+')
plt.axis('equal')
plt.grid()
plt.show()