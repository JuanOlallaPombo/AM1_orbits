from numpy import zeros, array
import matplotlib.pyplot as plt
from scipy.optimize import newton

#Declaración de variables iniciales

U_euler = array([1, 0, 0, 1])
U_cn = array([1, 0, 0, 1])
U_rk = array([1, 0, 0, 1])

#Partición temporal

N = 10000
dt = 0.001

#Función F del problema en diferencias finitas (atracción gravitatoria)
def F_Kepler(U):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x**2 + y**2)**1.5
    return array([vx, vy, -x/mr, -y/mr])


#Inicialización de variables
x_euler=array(zeros(N))
y_euler=array(zeros(N))
x_euler[0] = U_euler[0]
y_euler[0] = U_euler[1]

x_cn=array(zeros(N))
y_cn=array(zeros(N))
x_cn[0] = U_cn[0]
y_cn[0] = U_cn[1]

x_rk=array(zeros(N))
y_rk=array(zeros(N))
x_rk[0] = U_rk[0]
y_rk[0] = U_rk[1]

#Euler Method
for i in range(1,N):
    F = F_Kepler(U_euler)
    U_euler = U_euler + dt * F
    x_euler[i] = U_euler[0]
    y_euler[i] = U_euler[1]

#Crank-Nicolson Method
for i in range(1,N):
    F = F_Kepler(U_cn)
    def CN_Res(x):
        return x - U_temp - dt/2 * F_Kepler(x)      #Aplicamos Newton, definimos el residuo
    U_temp = U_cn + dt/2 * F_Kepler(U_cn)           #Inicializamos el sistema en Un porque va a estar muy cerca de Un1
    U_cn = newton(CN_Res, U_cn)
    x_cn[i] = U_cn[0]
    y_cn[i] = U_cn[1]

#Runge-Kutta 4th Order
for i in range(1,N):
    F = F_Kepler(U_rk)
    k1 = F
    k2 = F_Kepler(U_rk+0.5*dt*k1)
    k3 = F_Kepler(U_rk+0.5*dt*k2)
    k4 = F_Kepler(U_rk+dt*k3)
    U_rk = U_rk + dt/6*(k1+2*k2+2*k3+k4)
    x_rk[i] = U_rk[0]
    y_rk[i] = U_rk[1]

#Gráficas de resultados para comparar
plt.figure(1)
plt.title('Euler')
plt.axis('equal')
plt.plot(x_euler, y_euler)

plt.figure(2)
plt.title('Crank-Nicolson')
plt.axis('equal')
plt.plot(x_cn, y_cn)

plt.figure(3)
plt.title('Runge-Kutta 4')
plt.axis('equal')
plt.plot(x_rk, y_rk)
plt.show()