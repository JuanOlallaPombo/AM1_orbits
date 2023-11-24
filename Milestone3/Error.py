from Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler
from Cauchy_Problem import F_Cauchy
from numpy import zeros, array, linalg
from math import sqrt, log
import matplotlib.pyplot as plt

#Dominio de tiempo
t = 10
N1a = array(zeros(19))
N2a = array(zeros(19))
dt1a = array(zeros(19))
dt2a = array(zeros(19))
Error = array(zeros(19))

U_0 = array([1,0,0,1])

def F_Kepler(U, t):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])

for i in range(0,19):
    N1a [i] = int(1000+500*(i))
    N2a [i] = int(N1a[i]*2)
    dt1a [i] = t/N1a[i]
    dt2a [i] = t/N2a[i]

    N1b = int(N1a[i])
    N2b = int(N2a[i])

    U1= F_Cauchy(Euler, U_0, F_Kepler, t, dt1a[i])
    U2= F_Cauchy(Euler, U_0, F_Kepler, t, dt2a[i])

    U1error = U1[:,N1b-1]
    U2error = U2[:,N2b-1]

    Error [i] = log(linalg.norm(U1error-U2error))

q = (Error[18]-Error[0])/(log(N1a[0])-log(N1a[18]))
print("El orden del esquema es ",q)


#Definimos dos mallas, con una el doble de fina que la otra

N1=1000
N2=2*N1
dt1=t/N1
dt2=t/N2

U_0 = array([1,0,0,1])  #Setting initial conditions, starts in point (1,0) with vertical velocity

order=2
U1= F_Cauchy(Inverse_Euler, U_0, F_Kepler, t, dt1)
U2= F_Cauchy(Inverse_Euler, U_0, F_Kepler, t, dt2)

E=array(zeros(N1))

diff=0
for k in range(0,N1):
    for j in range (0,len(U_0)):
        diff=diff+(U1[j,k]-U2[j,2*k])**2
    E[k]=sqrt(diff)/(1-1/2**order)

Tiempo = [dt1*i for i in range(0,N1)]

plt.figure(1)
plt.title('Error')
plt.plot(Tiempo,E)
plt.show()




