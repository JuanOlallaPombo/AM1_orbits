from numpy import array, zeros, linspace, abs, transpose, float64
import matplotlib.pyplot as plt
from Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler

#Calculates and shows stability regions for each temporal scheme
def Stability_Region(Scheme, N, x0, xf, y0, yf):

    x, y = linspace(x0, xf, N), linspace(y0, yf, N)
    rho = zeros((N,N), dtype=float64)

    for i in range(N):
        for j in range(N):

            w=complex(x[i], y[j])
            r = Scheme (1., lambda u, t: w*u, 1., 0.)
            rho[i,j] = abs(r)

    return rho, x, y


#Function that test the previous function
def test_Stability_Regions():

    schemes = [Euler, Crank_Nicolson, RK4, Inverse_Euler]

    for scheme in schemes:
        rho, x, y = Stability_Region(scheme, 100, -4, 2, -4, 4)
        plt.contour(x, y, transpose(rho), linspace(0,1,11))
        plt.axis('equal')
        plt.grid()
        plt.show()

if __name__=='__main__':
    test_Stability_Regions()        #Garantiza que solo se hace la función test cuando el programa se corre como main, no al haber sido llamado por otro programa
