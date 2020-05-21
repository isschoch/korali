import tkinter as tk
from tkinter import *
import json

#FILES import
import classes_FRAMES
import class_KORALI
import functions

# Frame Variables:
selectorColor = 'snow'
forbidden = ['Variables','Problem','Solver']


class Problems():
    def __init__(self,master,experiments,directorio,nombre,DB,cont):
        # master is the frame from the previous class where we want to insert data.

        self.problem = tk.Frame(master,bg=selectorColor,width=717,height=925)
        self.problem.grid(column=0,row=0)
        self.problem.grid_propagate(0)

        experiments = classes_FRAMES.experiments
        selectedtab = classes_FRAMES.selectedtab

        # STORE THIS FRAME IN THE experiments dictionary.
        experiments[selectedtab]['problem'] = self.problem

        results = experiments[selectedtab]['results']
        results[cont] = {}

        functions.printConfig(self.problem,experiments,selectedtab,directorio,nombre,DB,cont)



