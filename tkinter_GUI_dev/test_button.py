# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:27:11 2018

@author: Chens
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk
import tkinter.ttk as ttk
import sys

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self,master)
        self.createWidgets()

    def createWidgets(self):
        fig=Figure()
        ax=fig.add_subplot(111 ,polar=True)
        canvas=FigureCanvasTkAgg(fig,master=root)
        canvas.get_tk_widget().grid(row=0,column=1)
        canvas.show()

        self.plotbutton=tk.Button(master=root, text="plot", command=lambda: self.plot(canvas,ax))
        self.plotbutton.grid(row=0,column=0)

    def plot(self,canvas,ax):
        c = ['r','b','g']  # plot marker colors
        ax.clear()         # clear axes from previous plot
        for i in range(3):
            theta = np.random.uniform(0,360,10)
            r = np.random.uniform(0,1,10)
            ax.plot(theta,r,linestyle="None",marker='o', color=c[i])
            canvas.draw()

root=tk.Tk()
app=Application(master=root)
app.mainloop()