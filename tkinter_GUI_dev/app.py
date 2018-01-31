# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:32:32 2018

@author: Chens
"""

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, colorchooser
from tkinter import ttk
from PIL import Image, ImageTk

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()




LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)

#---------------------------- Configurations--------------------
trLayout = [1, 33, 17, 29, 13, 93, 49, 81, 65, 77, 61, 21, 25, 9, 41, 5, 37,
            69, 73, 57, 89, 53, 85, 45, 2, 34, 18, 30, 14, 94, 50, 82, 66, 78,
            62, 22, 26, 10, 42, 6, 38, 70, 74, 58, 90, 54, 86, 46, 3, 35, 19, 
            31, 15, 95, 51, 83, 67, 79, 63, 23, 27, 11, 43, 7, 39, 71, 75, 59,
            91, 55, 87, 47, 4, 36, 20, 32, 16, 96, 52, 84, 68, 80, 64, 24, 28,
            12, 44, 8, 40, 72, 76, 60, 92, 56, 88, 48]

#trLayout = np.linspace(1, 96, 96, dtype = 'uint')
trLayout =  np.asarray(trLayout) - 1

#------------------------------Main Signal-------------------------------
MATRICES_SIZE = (96, 520, 2000)
signal_matrices = np.zeros(MATRICES_SIZE,  dtype='float16')

#-------------------------------Calliper---------------------------------
MATRIX_SIZE = (96, 520)
distance = np.zeros(MATRIX_SIZE,  dtype='float16')
START_DELAY = 6601




# since using imported pyplot

f = Figure()
signal_plot = f.add_subplot(221)
time_energy_plot = f.add_subplot(222)

calliper_plot = f.add_subplot(223)
thinkness_plot = f.add_subplot(224)

def processBinFile(OpenedFile):
    raw_data = np.fromfile(OpenedFile, dtype = np.uint8)
    bin_file_size = len(raw_data)  
    ii = np.zeros((1,128), dtype=np.int)
    start_byte = 0
    rp_i = 0
    rp_locs = np.zeros(6240, dtype='int')    
    for i in range(1, int(bin_file_size/32096) + 1):
        raw_fire_time = raw_data[start_byte + 24:start_byte + 32]
        roll_b = raw_data[start_byte + 16:start_byte + 18].view('int16')
        pitch_b = raw_data[start_byte + 18:start_byte + 20].view('int16')
        if((roll_b != 8224) | (pitch_b != 8224)):
            rp_locs[rp_i] = i
            rp_i = rp_i + 1
            
        for k in range(0, 8):
            raw_signal = raw_data[start_byte + k * 4008 + 40 : start_byte + k * 4008 + 4040].view('uint16')
            raw_signal = np.float16((raw_signal.astype("double")-32768)/32768)
            raw_signal = np.asmatrix(raw_signal)
            #raw_first_ref = raw_data[start_byte+k*4008+32:start_byte +k*4008+34]
            #first_ref = raw_first_ref.view('uint16')
            channel_index = raw_data[start_byte + k*4008 + 38].astype("int")
            # FUTURE : add thickness and distance calculation here to save more time
            signal_matrices[channel_index, ii[0,channel_index], :] = raw_signal
            ii[0,channel_index] = ii[0,channel_index] + 1
        start_byte = start_byte +32096
    return signal_matrices


def calliperMap(signal_matrices):
    TOTAL_CHN, TOTAL_ROUND, SIGNAL_LENGTH = signal_matrices.shape

    for chn in range(TOTAL_CHN):
        for rd in range(TOTAL_ROUND):
            signal = signal_matrices[trLayout[chn], rd, :]
            norm_signal = np.float16( signal/np.max(np.absolute(signal)))
            # USE Numpy.argmax instead of for loop to save time
            trigger = np.argmax(norm_signal > 0.594)
            if (trigger < 20) or (trigger > 1700):
                trigger = 20
            else:
                pass
#            main_reflection = norm_signal[trigger - 20 : trigger + 380]
#            main_signal_matrices[chn, rd, :] = main_reflection
            distance[chn,rd] = np.float32( (START_DELAY + trigger)*740.0/15000000)
    return distance
    
    
        
def OpenBinFile():
    name = filedialog.askopenfilename(initialdir="C:/Users/chens/Documents/gui-dev/SmallTempData",
                           filetypes =(("Binary File", "*.bin"),("All Files","*.*")),
                           title = "Choose a file."
                           )
    print (name)
    #Using try in case user types in unknown file or closes without choosing a file.
    try:
        with open(name,'rb') as OpenedFile:
            signal_matrices = processBinFile(OpenedFile)
            distance = calliperMap(signal_matrices)
    except:
        print("No file exists")

def showdialog():
    '''各种窗口'''

    res = simpledialog.askstring(title='字符串', prompt='输入一个字符串')

    print(res)
   



class BlueNoseApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.geometry('800x600')                  # 窗口大小
        tk.Tk.iconbitmap(self, 'example.ico')
        tk.Tk.wm_title(self, "BlueNose Signal Analyzer App")
#        self.createUI()
        self.createICO()
        self.createMenu()
        self.createToolbar()
        self.bindAll()
        
        container = tk.Frame(self, padx = 12, pady = 12)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Container for all different frames.. e.g. pages
        self.frames = {}
        
        #for F in (StartPage, PageOne, PageThree):
        for F in (StartPage, GraphMainPage):
            frame = F(container, self)
            self.frames[F] = frame
            
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        
        frame = self.frames[cont]
        # raise the fram to the front
        frame.tkraise()        
        
        
#------------- 生成界面 ----------------------
#    def createUI(self):
#        self.createICO()
#        self.createMenu()
#        self.createToolbar()
#        self.bindAll()
    
    # 创建菜单
    def createMenu(self):
        '''只支持两层嵌套'''
        menus = ['File', 'Edit', 'Help']
        items = [['New', 'Open', 'Save', 'Save as', 'Close', '-', 'Quit'],
                 ['Undo', '-',  'Cut', 'Copy',['更多...','数据', '图表', '统计']],
                 ['索引', '关于']]
        callbacks = [[showdialog, showdialog, showdialog, showdialog, showdialog, None, showdialog],
                     [showdialog, None, showdialog, showdialog, [showdialog, showdialog, showdialog]],
                     [showdialog, showdialog]]
        icos = [[self.img1, self.img2, self.img3, None, self.img4, None, None],
                [self.img1, None, self.img2, None, [self.img3, None, self.img4]],
                [None, None]]
        
        menubar = tk.Menu(self)
        for i,x in enumerate(menus):
            m = tk.Menu(menubar, tearoff=0)
            for item, callback, ico in zip(items[i], callbacks[i], icos[i]):
                if isinstance(item, list):
                    sm = tk.Menu(menubar, tearoff=0)
                    for subitem, subcallback, subico in zip(item[1:], callback, ico):
                        if subitem == '-':
                            sm.add_separator()
                        else:
                            sm.add_command(label=subitem, command=subcallback, image=subico, compound='left')
                    m.add_cascade(label=item[0], menu=sm)
                elif item == '-':
                    m.add_separator()
                else:
                    m.add_command(label=item, command=callback, image=ico, compound='left')
            menubar.add_cascade(label=x, menu=m)
        self.config(menu=menubar)
        
    # 生成所有需要的图标
    def createICO(self):
        #image_pil = Image.open('binary_ico.png').resize(16, 16)
        self.img1 = ImageTk.PhotoImage(Image.open('binary_file_icon.png'))
        self.img2 = ImageTk.PhotoImage(Image.open('ico_home.jpg'))
        self.img3 = ImageTk.PhotoImage(Image.open('ico_home.jpg'))
        self.img4 = ImageTk.PhotoImage(Image.open('ico_home.jpg'))
    
    # 生成工具条
    def createToolbar(self):
        toolframe = tk.Frame(self, height=20, bg='#F7EED6')#, relief=tk.RAISED)
        frame = tk.Frame(toolframe, bg='#F7EED6')
        ttk.Button(frame, width=20, image=self.img1, command=OpenBinFile).grid(row=0, column=0, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img2, command=showdialog).grid(row=0, column=1, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img3, command=showdialog).grid(row=0, column=2, padx=1, pady=1, sticky=tk.E)
        frame.pack(side=tk.LEFT)
        toolframe.pack(fill=tk.X)
        

    # 绑定快捷键
    def bindAll(self):
        self.bind_all('<Control-n>', lambda event:showdialog()) # 此处必须 lambda

"""
The start Page: 
"""        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # add text to window 
        label = tk.Label(self, text="""BlueNose Sound Signal Analyzer App, Version 0.0.1""", font=LARGE_FONT)
        label.pack(pady=10,padx=10) # adding paddings around to look neat
        # Define Button here
        button = ttk.Button(self, text="Go Analyzing~",
                            command=lambda: controller.show_frame(GraphMainPage))
        button.pack()

#        button_graph = ttk.Button(self, text="Go to Graph Page",
#                            command=lambda: controller.show_frame(PageThree))
#        button_graph.pack()
 

class GraphMainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # add text to window 
        label = tk.Label(self, text="Graph Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10) # adding paddings around to look neat        
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()   
        
        #--------------------Canvas for Plotting-------------------------
        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        
        
        
#    def draw(self):
#        timepoints = np.linspace(1, 2000, 2000, dtype = 'int')
#        signal_plot.clear()
#        signal_plot.plot(signal_matrices[0,0,:], timepoints, "g", label="signal")
#        title = "One channel Signal"
#        signal_plot.set_title(title)
#        sns.heatmap(distance, ax = calliper_plot)
#        self.canvas.draw()
#        
def animate(i):
    
    signal_plot.clear()
    calliper_plot.clear()
    time_energy_plot.clear()
    signal_plot.plot( signal_matrices[0,0,:], np.linspace(1, 2000, 2000, dtype = 'int'), "r-", label="signal")
    #a.plot_date(sellDates,sells["price"], "r", label="Normalized Magnitude")
    time_energy_plot.plot( signal_matrices[1,4,:], np.linspace(1, 2000, 2000, dtype = 'int'), "g", label="signal")
    #a.legned() #(bbox_to_anchor)
    #a.legend(bbox_to_anchor=(0,1.02,1,.102), loc=3,ncol=2,borderaxespad=0)

    calliper_plot.imshow(distance, extent=[0, 1, 0, 1], cbar = False, xticklabels = False, yticklabels = False)
    title = "One channel Signal"
    signal_plot.set_title(title)

       
app = BlueNoseApp()
ani = animation.FuncAnimation(f, animate, interval=10000)
app.mainloop()