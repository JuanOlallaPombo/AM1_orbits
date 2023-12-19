#Módulo para pasar de soluciones en X, Y, Z
#a frecuencias, ritmos y velocidades
from numpy import zeros,empty, asarray
from math import log

def Intervalos_Escala(x,E1,tau):
    '''
    x = solucion en x del sistema
    delta = division del tono, se supone que va de 0 a 0.5
    k = numero de octavas: 0<=k<7
    chi = estructura de la escala
    tau = tono musical
    o = octava -- cogemos 3
    '''
    k=7
    '''Generacion de Escala'''
    lamnbda= 1/2

    # p=int(6*k/lamnbda)

    # S=[]
    # E=[]
    # V=[]
    x1=zeros(len(x))

    # #Escala de intervalos
    # for i in range(0,p):
    #     S.append(2**(i*lamnbda/6))

    # ##hallar el vector binario V
    # for i in range(0,p):
    #     if chi[i] in S:         ##Aqui va la estructura de la escala tonal del cual no se el formato 
    #         V.append(1)
    #     else:
    #         V.append(0)

    # for i in range(0,p):

    #     E.append(S[i]*V[i])
    
    # E1 = list(filter(lambda x: x != 0, E))

    '''Normalizacion de la variable'''
    alpha = (2**k-1)/(max(x)-min(x))
    beta = -alpha*min(x)+1

    for i in range(0,len(x)):
        x1[i] = alpha*x[i]+beta

    '''Mapeo al valor mas proximo'''
    mu=2**(lamnbda/6)-1
    D=empty((len(x1),len(E1)))
    for i in range(0,len(x1)):
        for j in range (0,len(E1)):
            if abs(x1[i]-E1[j])<=mu:
                D[i,j]=0
            else:
                D[i,j]=x1[i]
    print("D = ", D)
    L=[]
    for i in range(0,len(x1)):
        L.append(min(D[i,:]))

    f=55*2**((tau+12*0-10)/12)

    '''Frecuencias de las notas'''
    F=[]
    for i in range(0,len(x1)):
        F.append(f*L[i])
    print("F = ", F)

    '''valores de la frecuencia para MIDI'''
    theta=[]
    for i in range(0,len(F)):
        if abs(F[i]) / 440 <= 0:
            theta.append(69)
        else:
            theta.append(69 + 12 / log(2) * log(abs(F[i]) / 440))
    theta2=[]
    for i in range(0,len(theta)):
        theta2.append(round(theta[i]))
    #print(y2)
    return asarray(theta2)


def Ritmo(y,tp):
    '''
    y = solucion y del sistema
    tp = tiempo musical
    '''

    R=[0,1,2,3,4,5,6]
    '''
    0 - semifusa
    1 - fusa
    2 - semichorchea
    3 - corchea
    4 - negra
    5 - blanca
    6 - redonda
    '''
    '''Normalizacion de la variable'''
    alpha = (max(R)-min(R))/(max(y)-min(y))
    beta = -alpha*min(y)+min(R)
    y1 = zeros(len(y))
    #print("y output = ", y)

    for i in range(0,len(y)):
        y1[i] = alpha*y[i]+beta
    #print(y1)

    '''Redondeo a valor proximo'''
    y2=[]
    for i in range(0,len(y)):
        y2.append(round(y1[i]))
    #print(y2)

    '''Conversion a MIDI'''
    Y=[]
    for i in range(0,len(y2)):
        Y.append(60/tp*2**y2[i]/2**4)

    return asarray(Y)

def Dinamica(z):
    '''
    z = solucion del sistema z
    '''

    U=[10, 30, 45, 60, 75, 92, 108, 127]
    '''
    0-10 ppp
    11-30 pp
    31-45 p
    46-60 mp
    61-75 mf
    76-92 f
    93-108 ff
    109-127 fff
    '''

    '''Normalizacion de la variable'''
    alpha = (max(U)-min(U))/(max(z)-min(z))
    beta = -alpha*min(z)+min(U)
    z1 = zeros(len(z))

    for i in range(0,len(z)):
        z1[i] = alpha*z[i]+beta

    '''Mapeo al valor mas proximo'''
    mu=10
    D=empty((len(z),len(U)))
    for i in range(0,len(z)):
        for j in range (0,len(U)):
            if abs(z[i]-U[j])<=mu:
                D[i,j]=0
            else:
                D[i,j]=z[i]

    L=[]
    for i in range(0,len(z)):
        L.append(min(D[i,:]))

    Z=[]
    for i in range(0,len(L)):
        if max(L) != 0:
            Z.append(L[i]/max(L))
        else:
            Z.append(100)

    return asarray(Z)
