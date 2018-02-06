# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:22:07 2018

@author: Chens
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:31:11 2018

@author: Chens
"""
import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, colorchooser
from tkinter import ttk
from PIL import Image, ImageTk

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import pickle
from scipy.signal import hilbert

LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)


INITIAL_DIR = os.getcwd()
## GLOBAL VARIABLE
InputFolderPath_G =  INITIAL_DIR
OutputFolderPath_G = INITIAL_DIR
#### Layout part
trLayout = [1, 33, 17, 29, 13, 93, 49, 81, 65, 77, 61, 21, 25, 9, 41, 5, 37,
            69, 73, 57, 89, 53, 85, 45, 2, 34, 18, 30, 14, 94, 50, 82, 66, 78,
            62, 22, 26, 10, 42, 6, 38, 70, 74, 58, 90, 54, 86, 46, 3, 35, 19, 
            31, 15, 95, 51, 83, 67, 79, 63, 23, 27, 11, 43, 7, 39, 71, 75, 59,
            91, 55, 87, 47, 4, 36, 20, 32, 16, 96, 52, 84, 68, 80, 64, 24, 28,
            12, 44, 8, 40, 72, 76, 60, 92, 56, 88, 48]
LAYOUT1 =  np.asarray(trLayout) - 1
trLayout = np.linspace(1, 96, 96, dtype = 'uint')
LAYOUT2 =  np.asarray(trLayout) - 1
LAYOUT3 = LAYOUT2
TR_LAYOUT = (LAYOUT1, LAYOUT2, LAYOUT3)

### SIGNALs
S = [-0.0729,   -0.2975,   -0.2346 ,   0.1057   , 0.8121  ,  0.5721  , -0.4512,   
     -0.7820  , -0.5137    , 0.4829    ,0.8867 ,  -0.0891 ,  -0.4474  ,-0.0875 ,   0.2159]

START_DELAY = 6601
MATRIX_SIZE = (96, 520)
MATRICES_SIZE = (96, 520, 2000)

#------------------- For positioning and lining--------------------------
TRANSDUCER_PING_RATE = 104.1667
POS_ARRAY = [0, 55.88, 111.76, 27.94, 83.82, 139.7]
TOOL_SPEED = 0.396

TIME_FLIGHT = 50


#------
POS_ARRAY = np.asarray(POS_ARRAY)/1000/TOOL_SPEED * TRANSDUCER_PING_RATE
POS_ARRAY = np.round(POS_ARRAY).astype(int)
NumOfSampleOffsetPrChn = np.tile(POS_ARRAY, 16)

# --------------------GLOBAL VARIABLES
TRIGGER_MAP = np.zeros(MATRIX_SIZE,  dtype='uint16')
SIGNAL_MATRICES = np.zeros(MATRICES_SIZE,  dtype='float16')
NORM_SIGNAL_MATRICES = np.zeros(MATRICES_SIZE,  dtype='float16')
thickness_map = np.zeros(MATRIX_SIZE,  dtype='float64')



def processBinFile(OpenedFile):
    """
    processing the one opened bin file
    @return: signal matrices and roll_r
    """
    raw_data = np.fromfile(OpenedFile, dtype = np.uint8)
    bin_file_size = len(raw_data)  
    ii = np.zeros((1,128), dtype=np.int)
    start_byte = 0
    rp_i = 0
    rp_locs = np.zeros(6240, dtype='int') 
    roll_r = np.zeros(( 260 , 1 ), dtype=np.uint16)
    for i in range(1, int(bin_file_size/32096) + 1):
        raw_fire_time = raw_data[start_byte + 24:start_byte + 32]
        roll_b = raw_data[start_byte + 16:start_byte + 18].view('int16')
        pitch_b = raw_data[start_byte + 18:start_byte + 20].view('int16')
        if((roll_b != 8224) | (pitch_b != 8224)):
            rp_locs[rp_i] = i
            roll_r[rp_i] = roll_b
            rp_i = rp_i + 1
            
        for k in range(0, 8):
            raw_signal = raw_data[start_byte + k * 4008 + 40 : start_byte + k * 4008 + 4040].view('uint16')
            raw_signal = np.float16((raw_signal.astype("double")-32768)/32768)
            raw_signal = np.asmatrix(raw_signal)
            #raw_first_ref = raw_data[start_byte+k*4008+32:start_byte +k*4008+34]
            #first_ref = raw_first_ref.view('uint16')
            channel_index = raw_data[start_byte + k*4008 + 38].astype("int")
            SIGNAL_MATRICES[channel_index, ii[0,channel_index], :] = raw_signal
            ii[0,channel_index] = ii[0,channel_index] + 1
        start_byte = start_byte +32096
    return SIGNAL_MATRICES, roll_r


def processRoll_r (roll_r):
    """
    Processing roll_r
    @ return: 
    """
    tdcTr1 = 152

    TDC = roll_r /100+tdcTr1
    
    # Calculate transducer number @ TDC:
    transducer_TDC = TDC // 3.75 #np.around((TDC/3.75))
    
    transducer_TDC[transducer_TDC < 0] = transducer_TDC[transducer_TDC < 0] + 96
    transducer_TDC[transducer_TDC > 95] = transducer_TDC[transducer_TDC > 95] - 96
    
    transducer_TDC = np.interp(np.linspace(0, 519, 520, dtype = 'uint16'), 
                               np.linspace(0, 259, 260, dtype = 'uint16'), 
                               np.reshape(transducer_TDC, len(transducer_TDC)))
    return transducer_TDC

def take_3D_norm(signal_matrices):
    """
    Take 3-D matrices and 
    @ return a normalized 3-D matrix
    """
    TOTAL_CHN, TOTAL_ROUND, SIGNAL_LENGTH = signal_matrices.shape
    for chn in range(TOTAL_CHN):
        for rd in range(TOTAL_ROUND):
            signal = signal_matrices[trLayout[chn], rd, :]
            norm_signal = np.float16( signal/np.max(np.absolute(signal)))
            NORM_SIGNAL_MATRICES[chn, rd, :] = norm_signal
    return NORM_SIGNAL_MATRICES

def calculate_trigger_map(norm_matrices):
    """
    Calculate the trigger_map/calliper map from normalized 3-D matrix
    @input: normalized 3-D matrix
    @return: trigger map (2-D matrix)
    """
    TOTAL_CHN, TOTAL_ROUND, SIGNAL_LENGTH = norm_matrices.shape
    for chn in range(TOTAL_CHN):
        for rd in range(TOTAL_ROUND):
            norm_signal = norm_matrices[trLayout[chn], rd, :]
            trigger = np.argmax((np.absolute(norm_signal) > 0.594)) # use Absolute in case the negative edge coming first (Phase)
            if (trigger < 50):
                trigger = 50
            elif (trigger > 1499):
                trigger = 1499
            else:
                pass
            TRIGGER_MAP[chn, rd] = trigger
    return TRIGGER_MAP




#-------------------------------------------------------------------------------
def OpenBinFile():
    name = filedialog.askopenfilename(initialdir = INITIAL_DIR,
                           filetypes =(("Binary File", "*.bin"),("All Files","*.*")),
                           title = "Choose a file.")
    print (name)
    #Using try in case user types in unknown file or closes without choosing a file.
    try:
        with open(name,'rb') as OpenedFile:
            SIGNAL_MATRICES, roll_r = processBinFile(OpenedFile)
            NORM_SIGNAL_MATRICES = take_3D_norm(SIGNAL_MATRICES)
#            TRIGGER_MAP = calculate_trigger_map(NORM_SIGNAL_MATRICES)
#            trigger_map.dump("data_trigger_map_" + name[-19:-4]  - ".bin")
#            signal_matrices.dump("data_NormSignal_matrices_" + name[-19:-4]  - ".bin")
#            roll_r.dump("roll_r_" + name[-19:-4] - ".bin")
            print("Done")
    except:
        print("No file exists")

def showdialog():
    '''各种窗口'''

    res = simpledialog.askstring(title='字符串', prompt='输入一个字符串')

    print(res)


class SettingInterface:
    def __init__(self):
        self.trLayout_sel_G = 0
        
    #def update(self):

setting_interface = SettingInterface()

class BlueNoseApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.geometry('1920x1080')                  # 窗口大小
        tk.Tk.iconbitmap(self, 'example.ico')
        tk.Tk.wm_title(self, "BlueNose Signal Analyzer App")
    #        self.createUI()
        self.createICO()
        self.createMenu()
        self.createToolbar()
        self.bindAll()
        
        container = tk.Frame(self, padx = 20, pady = 20)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
    
        # Container for all different frames.. e.g. pages
        self.frames = {}
        
        for F in (WelcomePage, CalliperPage, TimeDashBoard):
            frame = F(container, self)
            self.frames[F] = frame
            
            frame.grid(row=0, column=0, sticky="nsew")
    
        self.show_frame(WelcomePage)
        
    def show_frame(self, cont):
        
        frame = self.frames[cont]
        # raise the fram to the front
        frame.tkraise()          
    
    
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
The WelcomPage Page: 
"""        
class WelcomePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.initVariable(parent, controller)
        self.createWidgets(parent, controller)
        
    
    def initVariable(self, parent, controller):
        self.InputFolderPath =  tk.StringVar() 
        self.InputFolderPath.set(InputFolderPath_G)
        self.OutputFolderPath = tk.StringVar()
        self.OutputFolderPath.set(OutputFolderPath_G)
    
    def createWidgets(self, parent, controller):
        # add text to window 
        label = tk.Label(self, text="""BlueNose Sound Signal Analyzer App, Version 1.0.1""", font=LARGE_FONT)
        label.grid(row=0, column=0, columnspan = 7, padx=10, pady=10, sticky='EW') 
        ttk.Separator(self, orient= 'horizontal').grid(row = 1, column = 0, columnspan = 7, sticky="ew",  padx=10, pady=10)
        tk.Label(self, text = "Single Bin File Analysis: ").grid(row = 2, column = 0, padx = 10 , pady = 10)
        tk.Label(self, text = "Folder Path").grid(row = 3, column = 0, padx = 10 , pady = 10)
        tk.Label(self, text = "Saving Path").grid(row = 4, column = 0, padx = 10 , pady = 10)
        tk.Label(self, text = "One Min Matrix File").grid(row = 5, column = 0, padx = 10 , pady = 10)
        
        # Single Bin File Analysis
        ttk.Button(self, text="Calliper & Movement Examine", command = lambda: controller.show_frame(CalliperPage)).grid(row = 2, column = 1, padx = 10 , pady = 10)
        ttk.Button(self, text="Single Channel Dashboard", command =lambda: controller.show_frame(TimeDashBoard)).grid(row = 2, column = 2, padx = 10 , pady = 10)   
        ttk.Button(self, text="MultiChannel Dashboard", command =showdialog).grid(row = 2, column = 3, padx = 10 , pady = 10)

        # Processing Folder
        tk.Entry( self, textvariable = self.InputFolderPath , width = 65).grid(row=3, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.OutputFolderPath,  width = 65 ).grid(row=4, column=1, padx=10, pady=10)
        ttk.Button(self, text="Processing Folder", command = showdialog).grid(row = 3, column = 2, rowspan = 2, padx = 10 , pady = 10)

        # Multiple Bin File Analysis
        ttk.Button(self, text="Project Settings", command =showdialog).grid(row = 5, column = 1, padx = 10 , pady = 10)
        ttk.Button(self, text="Dashboard", command =showdialog).grid(row = 5, column = 2, padx = 10 , pady = 10)   
        ttk.Button(self, text="Tests", command =showdialog).grid(row = 5, column = 3, padx = 10 , pady = 10)        

class CalliperPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.initVariable(parent, controller, setting_interface)        
        self.createWidgets(parent, controller)
        self.createPlots(parent, controller)
    
    def initVariable(self, parent, controller, interface):
        self.trLayout = tk.StringVar()
        self.trLayout.set(str(TR_LAYOUT[setting_interface.trLayout_sel_G]))
        self.channel_no = tk.IntVar()
        self.round_no = tk.IntVar()
        self.channel_no.set(0)
        self.round_no.set(0)
        self.start_delay = tk.IntVar()
        self.start_delay.set(6601)
        
    def createWidgets(self, parent, controller):
        tk.Label(self, text= """Calliper & Positioning Dashboard""", font=LARGE_FONT).grid(row=0, column=0, columnspan = 7, padx=10, pady=10, sticky='EW')
        ttk.Separator(self, orient= 'horizontal').grid(row = 1, column = 0, columnspan = 8, sticky="ew",  padx=10, pady=10)
        tk.Label(self, text = "TrLayout").grid(row = 2, column = 0, padx = 10 , pady = 10)
        tr_layout_box = ttk.Combobox(self, textvariable = self.trLayout, width = 35, values = (str(LAYOUT1), str(LAYOUT2), str(LAYOUT3)) ) 
                                     #, postcommand = self.updtcblist(self, interface))
                                                                            
        tr_layout_box.grid(row = 2, column = 1, padx = 10 , pady = 10) #初始化  
        #trLayout_sel_G = tr_layout_box.current()
        
        
        tk.Label(self, text = "Channel").grid(row = 3, column = 0, padx = 10 , pady = 10)
        tk.Label(self, text = "Round").grid(row = 5, column = 0, padx = 10, pady = 10)
        tk.Entry( self, textvariable = self.channel_no ).grid(row=3, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.round_no ).grid(row=5, column=1, padx=10, pady=10)
        button_fw_chn = ttk.Button(self, text=">")  
        button_fw_chn.grid(row = 4, column = 1, padx = 10 , pady = 10)
        button_fw_rd = ttk.Button(self, text=">")  
        button_fw_rd.grid(row = 6, column = 1, padx = 10 , pady = 10)        

        button_bw_chn = ttk.Button(self, text="<")  
        button_bw_chn.grid(row = 4, column = 0, padx = 10 , pady = 10)
        button_bw_rd = ttk.Button(self, text="<")  
        button_bw_rd.grid(row = 6, column = 0, padx = 10 , pady = 10)                
        
        tk.Label(self, text = "Start Delay", font = NORM_FONT).grid(row = 8, column = 0, padx = 10, pady = 10) 
        tk.Entry( self, textvariable = self.start_delay ).grid(row=8, column=1, padx=10, pady=10)
        ttk.Button(self, text="Signal Plot").grid(row = 9, column = 0, columnspan = 2, rowspan = 1, padx = 10, pady = 10)
        ttk.Button(self, text="Filtered Calliper Map Plot").grid(row = 10, column = 0, columnspan = 2, rowspan = 1, padx = 10, pady = 10)
        ttk.Button(self, text="Save Map").grid(row = 13, column = 0, columnspan = 2, rowspan = 1, padx = 10, pady = 10)
        ttk.Button(self, text="Return To Home", command = lambda: controller.show_frame(WelcomePage)).grid(row = 14, column = 0, columnspan = 1, rowspan = 1, padx = 10, pady = 10)
    
    def createPlots(self, parent, controller):
        signal_plot = Figure(figsize = (4,3))
        signal_plot_ax = signal_plot.add_subplot(111)        
        canvas_signal_plot = FigureCanvasTkAgg(signal_plot, self)
        canvas_signal_plot._tkcanvas.grid(row=2, column = 2, padx=0, pady=0 , columnspan = 3, rowspan = 4)
        canvas_signal_plot.show()     

        offset_plot = Figure(figsize = (4,3))
        offset_plot_ax = offset_plot.add_subplot(111)        
        canvas_offset_plot = FigureCanvasTkAgg(offset_plot, self)
        canvas_offset_plot._tkcanvas.grid(row=2, column = 5, padx=0, pady=0 , columnspan = 3, rowspan = 4)
        canvas_offset_plot.show()          
        
        calliper_plot = Figure(figsize = (8,4))
        calliper_plot_ax = calliper_plot.add_subplot(111)        
        canvas_calliper_plot = FigureCanvasTkAgg(calliper_plot, self)
        canvas_calliper_plot._tkcanvas.grid(row=6, column = 2, padx=0, pady=10 , columnspan = 6, rowspan = 4)
        canvas_calliper_plot.show()            
        

class TimeDashBoard(tk.Frame):
    
    def __init__(self, parent, controller):    
        tk.Frame.__init__(self, parent)
        self.initVariable(parent, controller)
        self.createWidgets(parent, controller)
        self.createPlot(parent, controller)
        self.createAnalyser(parent, controller)
    def initVariable(self, parent, controller):
        self.channel_no = tk.IntVar()
        self.round_no = tk.IntVar()
        self.channel_no.set(0)
        self.round_no.set(0)
        self.mph = tk.DoubleVar()
        self.mph.set(0.08)
        self.mpd = tk.IntVar()
        self.mpd.set(22)
        self.low_margin = tk.IntVar()
        self.low_margin.set(0) 
        self.high_margin = tk.IntVar()
        self.high_margin.set(800)
        
        self.df_original = tk.StringVar()
        self.df_correlation = tk.StringVar()
        self.df_hilbert = tk.StringVar()
        self.median_original = tk.IntVar()
        self.median_correlation = tk.IntVar()
        self.median_hilbert = tk.IntVar()
        self.mean_original = tk.DoubleVar()
        self.mean_correlation = tk.DoubleVar()
        self.mean_hilbert = tk.DoubleVar()

        
    def createWidgets(self, parent, controller):
        # row 0
        tk.Label(self, text="Single ChannelTime Domain Analysis Dashboard", 
                 font=LARGE_FONT).grid(row=0, column = 0, columnspan = 10, padx=10, pady=10)
        ttk.Separator(self, orient= 'horizontal').grid(row = 1, column = 0, columnspan = 10, 
                     sticky="ew") 
        ttk.Separator(self, orient= 'horizontal').grid(row = 12, column = 0, columnspan = 10, 
                     sticky="ew")     
        ttk.Separator(self, orient= 'horizontal').grid(row = 18, column = 0, columnspan = 10, 
                     sticky="ew")  
        ## column 1
        tk.Label(self, text="Channel", 
                 font=NORM_FONT).grid(row=2, column = 0, columnspan = 1, padx=10, pady=10)
        tk.Label(self, text="Round", 
                 font=NORM_FONT).grid(row=4, column = 0, columnspan = 1, padx=10, pady=10)
        
        tk.Label(self, text="MPH", 
                 font=NORM_FONT).grid(row=6, column = 0, columnspan = 1, padx=10, pady=10)        
        tk.Label(self, text="MPD", 
                 font=NORM_FONT).grid(row=7, column = 0, columnspan = 1, padx=10, pady=10)            
        tk.Entry( self, textvariable = self.channel_no ).grid(row=2, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.round_no ).grid(row=4, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.mph ).grid(row=6, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.mpd ).grid(row=7, column=1, padx=10, pady=10)        
        tk.Label(self, text="Map Margin Setting", 
                 font=NORM_FONT).grid(row=8, column = 0, columnspan = 2, padx=10, pady=10)  
        tk.Entry( self, textvariable = self.low_margin ).grid(row=9, column=0, padx=10, pady=10)
        tk.Entry( self, textvariable = self.high_margin ).grid(row=9, column=1, padx=10, pady=10)   
        
        
    def createAnalyser(self, parent, controller):
        tk.Label(self, textvariable=self.df_original, 
                 font=SMALL_FONT).grid(row=6, column = 5, columnspan = 1, padx=0, pady=0) 
        tk.Label(self, textvariable=self.df_correlation, 
                 font=SMALL_FONT).grid(row=11, column = 2, columnspan = 1, padx=0, pady=0) 
        tk.Label(self, textvariable=self.df_hilbert, 
                 font=SMALL_FONT).grid(row=11, column = 5, columnspan = 1, padx=0, pady=0) 
        tk.Label(self, textvariable=self.median_original, 
                 font=SMALL_FONT).grid(row=6, column = 6, columnspan = 1, padx=0, pady=0) 
        tk.Label(self, textvariable=self.median_correlation, 
                 font=SMALL_FONT).grid(row=11, column = 3, columnspan = 1, padx=0, pady=0) 
        tk.Label(self, textvariable=self.median_hilbert, 
                 font=SMALL_FONT).grid(row=11, column = 6, columnspan = 1, padx=0, pady=0) 
        tk.Label(self, textvariable=self.mean_original, 
                 font=SMALL_FONT).grid(row=6, column = 7, columnspan = 1, padx=0, pady=0) 
        tk.Label(self, textvariable=self.mean_correlation, 
                 font=SMALL_FONT).grid(row=11, column = 4, columnspan = 1, padx=0, pady=0) 
        tk.Label(self, textvariable=self.mean_hilbert, 
                 font=SMALL_FONT).grid(row=11, column = 7, columnspan = 1, padx=0, pady=0) 
           
    def createPlot(self, parent, controller):
        ## Figure Part
        full_signal_plot = Figure(figsize = (4,2))
        full_signal_plot_ax = full_signal_plot.add_subplot(111)
        canvas_full_signal_plot = FigureCanvasTkAgg(full_signal_plot, self)
        canvas_full_signal_plot._tkcanvas.grid(row=2, column = 2, padx=10, pady=10 , columnspan = 3, rowspan = 4)
        canvas_full_signal_plot.show()
        
        fp_partial = Figure(figsize = (4,2))
        fp_partial_ax = fp_partial.add_subplot(111)
        canvas_fp_partial = FigureCanvasTkAgg(fp_partial, self)
        canvas_fp_partial._tkcanvas.grid(row=2, column = 5, padx=10, pady=10 , columnspan = 3, rowspan = 4)
        canvas_fp_partial.show()
        
        fp_corrlation = Figure(figsize = (4,2))
        fp_corrlation_ax = fp_corrlation.add_subplot(111)
        canvas_fp_corrlation = FigureCanvasTkAgg(fp_corrlation, self)
        canvas_fp_corrlation._tkcanvas.grid(row=7, column = 2, padx=10, pady=10 , columnspan = 3, rowspan = 4)
        canvas_fp_corrlation.show()
        
        fp_hilbert = Figure(figsize = (4,2))
        fp_hilbert_ax = fp_hilbert.add_subplot(111)
        canvas_fp_hilbert = FigureCanvasTkAgg(fp_hilbert, self)
        canvas_fp_hilbert._tkcanvas.grid(row=7, column = 5, padx=10, pady=10 , columnspan = 3, rowspan = 4)
        canvas_fp_hilbert.show() 
        
        signel_transducer_plot = Figure(figsize = (4,4.6))
        signel_transducer_plot_ax = signel_transducer_plot.add_subplot(111)        
        canvas_signel_transducer_plot = FigureCanvasTkAgg(signel_transducer_plot, self)
        canvas_signel_transducer_plot._tkcanvas.grid(row=2, column = 8, padx=20, pady=20 , columnspan = 2, rowspan = 10)
        canvas_signel_transducer_plot.show()         
        
        thickness_plot = Figure(figsize = (12,2.9))
        thickness_plot_ax = thickness_plot.add_subplot(111)        
        canvas_thickness_plot = FigureCanvasTkAgg(thickness_plot, self)
        canvas_thickness_plot._tkcanvas.grid(row=13, column = 2, padx=10, pady=10 , columnspan = 8, rowspan = 4)
        canvas_thickness_plot.show()          

app = BlueNoseApp()
#ani = animation.FuncAnimation(f, animate, interval=10000)
app.mainloop()           
             
 