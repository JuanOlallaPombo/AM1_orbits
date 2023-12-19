# GENERACIÓN DE MÚSICA A PARTIR DE MODELOS MATEMÁTICOS

## Autores del trabajo: 
 * Jorge Arbesú Nieto
 * David Calleja Merino
 * Sandra Domínguez Gómez
 * Juan Olalla Pombo
 * Sofía Mesón Pérez 

## Objetivo del trabajo

EL presente trabajo tiene como objetivo desarrollar una melodía musical mediante el empleo de modelos matemáticos caóticos.

## Estructura del repositorio

El trabajo se encuentra desglosado de la siguiente manera:

1) *Attractors.py*: En este módulo se encuentra todos las ODEs que se pueden seleccionar en la GUI: Duffing, Van Der Poel, Rayleigh, Lorentz y Rossler.
2) *Cauchy_Problem.py*: El problema de Cauchy con el que se resolver.
3) *File_creation.py*: Convierte los vectores [x,y,z] en un archivo reproducible con cualquier reproductor de musica con extensión .mid.
4) *Output_Transformation.py*: Recoge las soluciones del vector [x,y,], se normalizan acorde al lenguaje musical para, que pueda ser leído como melodía.
5) *Temporal_Schemes.py*: Incluye los esquemas temporales empleados en el trabajo: Euler, Euler Implícito, Runge Kutta y Cranck Nicolson.
6) *Figures*:  Almacena las imagenes de descripción.

![](/Figures/Diagrama.png)
