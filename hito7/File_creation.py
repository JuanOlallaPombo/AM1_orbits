#Módulo para crear los archivos con la música
## SE RECIBEN 3 VECTORES
# -*- coding: utf-8 -*-
from numpy import array, integer, zeros, float64, arange
import tkinter as tk
from tkinter import filedialog
from mido import MidiFile, MidiTrack, Message

def create_midi_file(output_file, vector_nota, vector_velocidad, vector_bpm, Instrumento):

    # Crear un archivo MIDI
    midi = MidiFile()

    # Agregar una pista al archivo MIDI
    track = MidiTrack()
    midi.tracks.append(track)

    # Configurar el canal y el instrumento (opcional)
    track.append(Message('program_change', program=Instrumento))  # Cambiar el instrumento 

    # Agregar mensajes MIDI a la pista (secuencia de notas)
    for nota, velocidad, bpm in zip(vector_nota, vector_velocidad, vector_bpm):
        tiempo_en_segundos =  bpm  # Calcular el tiempo en segundos a partir de bpm
        tiempo_en_tics = int(tiempo_en_segundos * midi.ticks_per_beat)  # Convertir a tics de tiempo MIDI
        track.append(Message('note_on', note=nota, velocity=velocidad, time=0))  # Encender la nota
        track.append(Message('note_off', note=nota, velocity=velocidad, time=tiempo_en_tics))  # Apagar la nota

    # Guardar el archivo MIDI
    midi.save(output_file)

def get_save_path():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal

    # Mostrar el diálogo para elegir la ubicación y el nombre del archivo
    file_path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])

    return file_path

if __name__ == "__main__":
    # Obtener la ruta de archivo deseada mediante una ventana GUI
    output_file = get_save_path()

    # Crear el archivo MIDI
    if output_file:
        create_midi_file(output_file, vector_nota, vector_velocidad, vector_bpm)
