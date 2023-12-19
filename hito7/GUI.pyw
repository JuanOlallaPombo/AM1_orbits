import tkinter as tk
from tkinter import ttk, Tk
import numpy as np
from Attractors import ODE_Duffing, ODE_Rayleigh, ODE_Van_Der_Poel, Lorentz, Rossler
from numpy import zeros, arange, zeros_like, interp, array
from Cauchy_Problem import F_Cauchy
from Output_Transformation import Intervalos_Escala, Ritmo, Dinamica
from File_creation import create_midi_file, get_save_path
import matplotlib.pyplot as plt


class InterfazMateMusica(tk.Tk):
    def __init__(self):
        super().__init__()

        # ----------------------------ASPECTO------------------------------------
        # 
        # self.configure(bg="White")  # Color de fondo (puedes utilizar codigos hexadecimales o nombres de colores)
        self.geometry("850x350")
        
        # ---------------------------TITULO DE APLICACION------------------------

        self.title("MUSIC MUSE")
        
        # ----------------------ATRIBUTOS DE ALMACENAMIENTO DE VARIABLES---------------------
        self.modelo_seleccionado = tk.StringVar()
        self.parametros_entries = {}
        self.parametros_numericos = {}
        self.esquema_seleccionado = tk.StringVar()
        self.tiempo_de_integracion = tk.Spinbox(self, from_=0, to=12, width=7)
        self.Estructura_escala = {}
        self.Tonica = tk.Spinbox(self, from_=0, to=12, width=7)
        self.Tempo = {}
        self.Instrumentos ={}
        self.frase_label = tk.Label(self, text="" )
        self.frase_label.grid(row=2, column=1, padx=5, pady=5)
        
        
        ######  Etiqueta 'MODELO MATEMATICO'
        modelo_mate_label = tk.Label(self, text="Modelo Matemático")
        modelo_mate_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        ########### Desplegable en el modelo matematico
        opciones = ["Duffing", "Van Der Poel", "Rayleigh", "Lorentz", "Rossler"]
        self.modelo_mate_combobox = ttk.Combobox(self, values=opciones)
        self.modelo_mate_combobox.grid(row=1, column=1, padx=5, pady=5)

        # Etiqueta descripcion modelo matematico
        self.frase_label = tk.Label(self, text="")
        self.frase_label.grid(row=2, column=1, padx=5, pady=5)

        # Asociar la opcion al modelo matematico
        self.modelo_mate_combobox.bind("<<ComboboxSelected>>", self.actualizar_frase)
        
        
        ######  Etiqueta 'ESQUEMA TEMPORAL'
        esque_tempo_label = tk.Label(self, text="Esquema temporal")
        esque_tempo_label.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        ##########  Desplegable en lugar de Campo de texto
        esquemas = ["Euler", "Inverse_Euler", "RK4", "Crank_Nicolson"]
        self.esque_tempo_combobox = ttk.Combobox(self, values=esquemas)
        self.esque_tempo_combobox.grid(row=1, column=3, padx=5, pady=5)



        ######  Etiqueta BOTON DE EJECUCION 
        salga_segun_label = tk.Label(
            self, text="Seleccione la música que desea crear"
        )
        salga_segun_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        #-----------------SELECCION DE PARAMETROS-------------------
        parametros_label = tk.Label(self, text="Introduce parámetros")
        parametros_label.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.W)

        # **Parametros matematicos**
        letras = ["a", "b", "c"]
        self.number_entries = []

        for i, letra in enumerate(letras):
            # Label para mostrar la letra
            label = tk.Label(self, text=f"{letra}")
            label.grid(row=i + 4, column=1, sticky=tk.W, padx=5, pady=2)

            # Entry para permitir al usuario ingresar números
            entry = tk.Entry(self)
            entry.grid(row=i + 4, column=1, padx=5, pady=2)
            self.number_entries.append(entry)
            
        # **Parametros musicales**
        parametros_musicales_label = tk.Label(
            self, text="Parámetros Musicales"
        )
        parametros_musicales_label.grid(
            row=2, column=3, columnspan=2, pady=5, sticky=tk.W
        )


        # **Parametros estructura de la escala**
        ttk.Label(self, text="Estructura de la escala").grid(
            row=6, column=3, sticky=tk.E, padx=5, pady=2
        )
        
        opciones_estructura_escala = [
            "Cromática",
            "Mayor",
            "Menor natural",
            "Menor armónica",
            "Menor",
        ]
        self.combobox_estructura_escala = ttk.Combobox(
            self, values=opciones_estructura_escala, width=15
        )
        self.combobox_estructura_escala.grid(
            row=6, column=4, sticky=tk.W, padx=5, pady=2
        )
        # **Parametros Tempo musical**
        ttk.Label(self, text = "Tempo musical").grid(
            row=8, column=3, sticky=tk.E, padx=5,pady=2)
        
        opciones_tempo_musical = [
            "Rapido",
            "Lento",
            "Suave"
        ]
        self.combobox_tempo_musical = ttk.Combobox(
            self, values=opciones_tempo_musical, width=15
        )
        self.combobox_tempo_musical.grid(
            row=8, column=4, sticky=tk.W, padx=5, pady=2
        )
        
        # **Parametros Nº de octavas**
        ttk.Label(self, text=" Numero de octavas").grid(
            row=4, column=3,sticky=tk.E, padx=15, pady=2, 
        )
        opciones_octavas = ["7"]
        self.combobox_octavas = ttk.Combobox(self, values=opciones_octavas, width=7)
        self.combobox_octavas.grid(row=4, column=4, sticky=tk.W, padx=5, pady=2)
        
        # **Parametros Paso de Tiempo**
        ttk.Label(self, text="Paso del tiempo").grid(
            row=7, column=3, sticky=tk.E, padx=5, pady=2
        )
        opciones_Pasotiempo = ["0.1", "0.01", "0.001"]
        self.combobox_Pasotiempo = ttk.Combobox(
            self, values=opciones_Pasotiempo, width=7
        )
        self.combobox_Pasotiempo.grid(row=7, column=4, sticky=tk.W, padx=5, pady=2)

        # ** Parametros Instrumentos**

        ttk.Label(self, text = "Instrumentos").grid(
            row=9, column=3, sticky=tk.E, padx=5,pady=2)
        
        opciones_instrumentos = [
            "Piano",
            "Guitarra Elec",
            "Ocarina"
        ]
        self.combobox_instrumentos = ttk.Combobox(
            self, values=opciones_instrumentos, width=15
        )
        self.combobox_instrumentos.grid(
            row=9, column=4, sticky=tk.W, padx=5, pady=2
        )
        
        # --------------------Diccionario para almacenar los contadores
        self.contadores = {}

        param_numerico = {
            "Tiempo de integración": (0, 60),
            "Numero de octavas": opciones_octavas,
            "Tonica (tono inicial)": (0, 12),
            "Estructura de la escala": opciones_estructura_escala,
            "Paso del tiempo": opciones_Pasotiempo,
            "Tempo musical": opciones_tempo_musical,
            "Instrumentos": opciones_instrumentos
        }

        for i, (param, value_range) in enumerate(param_numerico.items()):
            contador_label = ttk.Label(self, text=f"{param}  ")
            contador_label.grid(row=i + 3, column=3, sticky=tk.E, padx=5, pady=2)

            # Utilizar un Combobox en lugar de un Spinbox para la estructura de la escala
            if param == "Estructura de la escala":
                contador = self.combobox_estructura_escala
            elif param == "Numero de octavas":
                contador = self.combobox_octavas
            elif param == "Paso del tiempo":
                contador = self.combobox_Pasotiempo
            elif param == "Tempo musical":
                contador = self.combobox_tempo_musical
            elif param == "Instrumentos":
                contador =  self.combobox_instrumentos
            else:
                contador = tk.Spinbox(
                    self, from_=value_range[0], to=value_range[1], width=7
                )

            contador.grid(row=i + 3, column=4, sticky=tk.W, padx=5, pady=2)
            # Asociar cada termino con su respectivo contador
            self.contadores[param] = contador

        # ----------------Boton Ejecutar--------------------------
        ejecutar_button = tk.Button(
            self, text="Genera la melodía", command=self.ejecutar, width=10
        )
        ejecutar_button.grid(
            row=len(letras) + 7,
            column=1,
            columnspan=2,
            pady=10,
            sticky=tk.W + tk.E + tk.S + tk.N,
        )

        # Centrar la ventana
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        x_coordinate = (screen_width - window_width) // 2
        y_coordinate = (screen_height - window_height) // 2
        self.geometry(f"+{x_coordinate}+{y_coordinate}")

    # ----------------------ACTUALIZACION DE FRASE PARA EL MODELO MATEMATICO-------------------------
    def actualizar_frase(self, event):
        # Obtener la opción seleccionada
        opcion_seleccionada = self.modelo_mate_combobox.get()

        # Definir las frases correspondientes a cada opción
        frases = {
            "Duffing": "Ecuación: x'' + ax'+ x^3= b*cos(t) ",
            "Van Der Poel": "Ecuación:  x'' + a*(x^2-1)*x' + b*x =0",
            "Rayleigh": "Ecuación: x''= x - x^3 - x'+ a*cos(t)",
            "Lorentz": "Ecuación: x'= a*(-x+y);  "   "y'= x'- x*z - y;  "  "z'= -b*z +x*y",
            "Rossler": "Ecuación: x' = -(y+z); y' = a*y + x; z' = b +z*(x-c)",
        }

        # Actualizar el texto de la etiqueta con la frase correspondiente
        self.frase_label.config(text=frases.get(opcion_seleccionada, ""))    
    

    #--------------------ALMACENAMIENTO DE VARIABLES-----------------------
    def ejecutar(self):

        # Obtener el modelo seleccionado
        modelo_seleccionado = self.modelo_mate_combobox.get()

        # Obtener los parámetros de letras
        parametros_entries = [entry.get() for entry in self.number_entries]
     
        abc = [float(valor) for valor in parametros_entries]
        a = abc[0]
        b = abc[1]
        c = abc[2]
        print(a)
        print(b)
        print(c)
        # Obtener el esquema temporal seleccionado
        esquema_seleccionado = str(self.esque_tempo_combobox.get())

        # Obtener los parámetros numéricos
        parametros_numericos = {}
        for param, contador in self.contadores.items():
            parametros_numericos[param] = contador.get()

        # Obtener tonica
        tiempo_integracion = self.contadores["Tiempo de integración"].get()

        # Obtener estructura de la escala
        Estructura_escala = self.combobox_estructura_escala.get()

        # Obtener tonica
        Tonica = self.contadores["Tonica (tono inicial)"].get()

        #Obtener tempo musical
        Tempo = self.contadores["Tempo musical"].get()
        
        #Obtener el instrumento musical 
        Instrumentos = self.contadores["Instrumentos"].get()
        
        # Imprimir la información
        print("Modelo seleccionado:", modelo_seleccionado)
        print("Parámetros de letras:", parametros_entries)
        print("Esquema temporal seleccionado:", esquema_seleccionado)
        print("Parámetros numéricos:", parametros_numericos)
        print("tiempo_integracion:", tiempo_integracion)
        print("Estructura de la escala:", Estructura_escala)
        print("Tonica:", Tonica)
        print("Tempo musical", Tempo)
        print("Instrumentos", Instrumentos)
        
        self.sample_rate = float(tiempo_integracion)
        dt = float(parametros_numericos["Paso del tiempo"])
        N = int(self.sample_rate/dt)
        self.t = arange(0, self.sample_rate, dt)
        self.x = zeros_like(self.t)
        self.y = zeros_like(self.t)
        self.y_time = zeros_like(self.t)
        
        ##-----SELECCION PARA INTRODUCIR PARAMETROS SEGUN MODELO MATEMATICO---
        if modelo_seleccionado == "Lorentz":
            U_0 = array([1, 2, 3])
            U = F_Cauchy(esquema_seleccionado, U_0, Lorentz, self.sample_rate, dt, a, b, c)

        elif modelo_seleccionado == "Rossler":

            model = Rossler
            U_0 = array([1, 2, 3])
            U = F_Cauchy(esquema_seleccionado, U_0, model, self.sample_rate, dt, a, b, c)

        elif modelo_seleccionado == "Duffing":

            model = ODE_Duffing
            U_0 = array([0, 0.75])
            U = F_Cauchy(esquema_seleccionado, U_0, model, self.sample_rate, dt, a, b, c)

        elif modelo_seleccionado == "Van Der Poel":

            model = ODE_Van_Der_Poel
            U_0 = array([1, 2])
            U = F_Cauchy(esquema_seleccionado, U_0, model, self.sample_rate, dt, a, b, c)

        elif modelo_seleccionado == "Rayleigh":

            model = ODE_Rayleigh
            U_0 = array([1, 2])
            U = F_Cauchy(esquema_seleccionado, U_0, model, self.sample_rate, dt, a, b, c)

        V = zeros((12, 1))
        V_completo = zeros((84, 1))
        S = zeros((84, 1))
        ##----ESCALA DE MUSICA-------------
        if Estructura_escala == "Cromática":
            V = array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        elif Estructura_escala == "Mayor":
            V = array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        elif Estructura_escala == "Menor natural":
            V = array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        elif Estructura_escala == "Menor armónica":
            V = array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
        elif Estructura_escala == "Menor melódica":
            V = array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        for i in range(0, 8):
            for j in range(0, 12):

                if j < 12 - int(Tonica):
                    V_completo[7 * i + j] = V[j - 12 + int(Tonica)]

                else:
                    V_completo[7 * i + j] = V[j - int(Tonica)]

        for i in range(0, 84):
            S[i] = 2 ** ((i + 1) / 12)

        self.E = V * S
        self.Notas = self.E[self.E != 0]

        self.x = U[0, :]
        #print("X sistema = ",self.x)
        self.y = U[1, :]
        #print("Y sistema = ",self.y)
        plt.figure(1)
        plt.title('Trajectory')
        plt.axis('equal')
        plt.plot(self.x,self.y)
        plt.show()

        if modelo_seleccionado == "Lorentz" or modelo_seleccionado == "Rossler":
            self.z = U[2, :]
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.set_title('Trajectory')
            ax1.plot(self.x,self.y,self.z,'g')
            ax1.set_aspect('auto')
            plt.show()
            #print("Z sistema = ",self.z)
            #z_normalized = interp(self.z, (self.z.min(), self.z.max()), (0, 127))

        self.x_final = Intervalos_Escala(self.x,self.Notas,int(Tonica))
        print("X final = ",self.x_final)
        tp = 60
        self.y_final = Ritmo(self.y,tp)
        print("Y final = ",self.y_final)
        if modelo_seleccionado == "Lorentz" or modelo_seleccionado == "Rossler":
            self.z_final = Dinamica(self.z)
        else:
            self.z_final = zeros(len(self.x), dtype=int)
            for i in range(0,len(self.z_final)):
                if Tempo=="Rapido":
                    self.z_final[i] = round(120)
                if Tempo=="Suave":
                    self.z_final[i] = round(90)
                if Tempo=="Lento":
                    self.z_final[i] = round(60)
        print("Z final = ",self.z_final)
     
            
        # Seleccionar instrumento
        if Instrumentos == "Piano":
            Instrumento = 0
        elif Instrumentos == "Guitarra Elec":
            Instrumento = 27
        elif Instrumentos == "Ocarina":
            Instrumento = 79  
        
        print("Numero de instrumento:", Instrumento)
        
        # Obtener la ruta de archivo deseada mediante una ventana GUI
        output_file = get_save_path()

        # Crear el archivo MIDI
        if output_file:
            create_midi_file(output_file, self.x_final, self.z_final, self.y_final,Instrumento)

        


if __name__ == "__main__":
    app = InterfazMateMusica()
    app.mainloop()
