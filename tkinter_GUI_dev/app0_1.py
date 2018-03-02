# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:31:11 2018

@author: Chens
"""

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, colorchooser
from tkinter import ttk
from PIL import Image, ImageTk

import numpy as np
import glob


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import pickle
from scipy.signal import hilbert
import scipy.signal
import os


INITIAL_DIR_TEST = os.getcwd()
## GLOBAL VARIABLE
InputFolderPath_G =  INITIAL_DIR_TEST
OutputFolderPath_G = INITIAL_DIR_TEST

# GUI CONFIGURE
INITIAL_DIR = "C:\\Users\\chens\\Documents\\gui-dev\\SmallTempData"
LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)

#-----------------------DATA PROCESSING-- Configurations--------------------
trLayout_bluenose = [1, 33, 17, 29, 13, 93, 49, 81, 65, 77, 61, 21, 25, 9, 41, 5, 37,
            69, 73, 57, 89, 53, 85, 45, 2, 34, 18, 30, 14, 94, 50, 82, 66, 78,
            62, 22, 26, 10, 42, 6, 38, 70, 74, 58, 90, 54, 86, 46, 3, 35, 19, 
            31, 15, 95, 51, 83, 67, 79, 63, 23, 27, 11, 43, 7, 39, 71, 75, 59,
            91, 55, 87, 47, 4, 36, 20, 32, 16, 96, 52, 84, 68, 80, 64, 24, 28,
            12, 44, 8, 40, 72, 76, 60, 92, 56, 88, 48]

trLayout_fraudhover = [1,17,13, 33, 5 ,93, 49, 65, 61, 81, 77, 21, 25, 41, 37, 9,29 , 69, 73, 89, 85, 57,
 53, 45, 2 ,18, 14, 34, 6 ,94, 50, 66, 62, 82, 78 ,22, 26, 42, 38, 10, 30, 70, 74,
 90, 86, 58, 54, 46, 3, 19, 15, 35, 7 ,95, 51, 67, 63, 83, 79, 23, 27 ,43, 39, 11,
 31, 71, 75, 91, 87, 59, 55, 47 ,4 ,20, 16, 36, 8 ,96, 52, 68, 64, 84, 80, 24, 28,
 44 ,40, 12, 32, 72, 76, 92, 88, 60, 56, 48]

trLayout = np.linspace(1, 96, 96, dtype = 'uint')
trLayout =  np.asarray(trLayout) - 1
LAYOUT1 = trLayout
#trLayout = np.linspace(1, 96, 96, dtype = 'uint')
LAYOUT2 =  np.asarray(trLayout_bluenose) - 1
LAYOUT3 = np.asarray(trLayout_fraudhover) - 1
TR_LAYOUT = (LAYOUT1, LAYOUT2, LAYOUT3)
TR_LAYOUT_NAME = ('NORMAL', 'BLUENOSE' , 'FraudHover')

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
CALLIPER_MAP= np.zeros(MATRIX_SIZE,  dtype='float')


thickness_map = np.zeros(MATRIX_SIZE,  dtype='float64')



ROLL_R = np.zeros(( 260 , 1 ), dtype=np.uint16)
# since using imported pyplot

trigger_map_f = Figure() # trigger map
trigger_map_plot = trigger_map_f.add_subplot(111)



f = Figure()
signal_plot = f.add_subplot(221)
time_energy_plot = f.add_subplot(222)
thinkness_plot = f.add_subplot(212)


def Dir_Parser(test_path):
    """
    Input: 
        @ test_path: a string containing the path of the directory
    Return: 
        @ file_map: a dictionary with the one min files
            key is the bnYearMonthDate-HourMin
            item is a list containing the file names in that min
    """
    if os.path.exists(test_path):
        os.chdir(test_path)
        currentPath = os.getcwd()
        print(currentPath)
    
    ## Filter .bin file and sort based on name
    file_list =  glob.glob('*.bin')
    file_list.sort()
    first_min = file_list[0][9:13]
    last_min = file_list[-1][9:13]
    
    ## Construct a dictionary for mapping mins of data 
    file_map = dict()
    for file in  file_list:
        if file[:-6] in file_map:
            pass
        else:
            bn_YMDHM = file[:-6]
            file_map[bn_YMDHM] = [f for f in file_list if f.startswith(bn_YMDHM)]   
            
    if len(file_map) > 1:
        sorted_key_list = sorted(file_map.keys())
        for key, index in  zip(sorted_key_list, range(len(sorted_key_list))):
            if key[9:13] < last_min:
                file_map[key].append(file_map[sorted_key_list[index + 1]][0])
            if key[9:13] > first_min: 
                file_map[key].insert(0,file_map[sorted_key_list[index - 1]][-1])
    print(file_map)            
    return file_map



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
    for i in range(1, int(bin_file_size/32096) + 1):
        raw_fire_time = raw_data[start_byte + 24:start_byte + 32]
        roll_b = raw_data[start_byte + 16:start_byte + 18].view('int16')
        pitch_b = raw_data[start_byte + 18:start_byte + 20].view('int16')
        if((roll_b != 8224) | (pitch_b != 8224)):
            rp_locs[rp_i] = i
            ROLL_R[rp_i] = roll_b
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
    return SIGNAL_MATRICES, ROLL_R

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

def calculate_trigger_map(norm_matrices, trLayout = trLayout):
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

def median_filter_2D (trigger_map, FILTER_DIM = 2):
    """
    Use 2-D median filter to smooth the trigger map
    """
    
    top_row = trigger_map[0 : FILTER_DIM, :]
    bot_row = trigger_map[-FILTER_DIM : , :]
    var_map1 =  np.vstack((trigger_map,top_row))
    var_map2 =  np.vstack((bot_row,var_map1))
    left_col = var_map2[:, 0 : FILTER_DIM]
    right_col = var_map2[:, -FILTER_DIM : ]    
    var_map3 =  np.hstack((var_map2,right_col))
    var_map4 =  np.hstack((left_col,var_map3))
    var_map5 = scipy.signal.medfilt(var_map4)
    final_map = np.int16(var_map5[FILTER_DIM: -FILTER_DIM, FILTER_DIM:-FILTER_DIM])
    
    return final_map

def calculate_calliper(trigger_map, start_delay = START_DELAY):
    TOTAL_CHN, TOTAL_ROUND = trigger_map.shape
    for chn in range(TOTAL_CHN):
        for rd in range(TOTAL_ROUND):
            CALLIPER_MAP[chn,rd] = np.float64( (start_delay + trigger_map[chn,rd])*740.0/15000000)
    return CALLIPER_MAP

def find_offset(TDC, chn, rd, calliper = CALLIPER_MAP, diameter = 1):
    """
    @ return:
        - x_offset at given round and chn
        - y_offset at given round and chn
    """
    top_ch_no = int(TDC[rd])
    bot_ch_no = (top_ch_no + 48)%96
    right_ch_no = (top_ch_no + 24)%96
    left_ch_no = (top_ch_no + 72)%96
    top_distance = calliper[top_ch_no, rd]
    bot_distance = calliper[bot_ch_no, rd]
    left_distance = calliper[left_ch_no, rd]
    right_distance = calliper[right_ch_no, rd]
    
    y_offset = (bot_distance - top_distance)/2/diameter
    x_offset = (left_distance - right_distance)/2/diameter
    return x_offset, y_offset

def find_envelope(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope




def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


def Thickness_map_TIME(signal_matrices):
    """
    1. convolution with S
    2. take upper envelope of the conv result
    3. detect peak of the envelope
    4. if len of diff(peak_loc) > 2:
            thickness = median(...)
            if thickness < 1.2*norm_thickness:
                Got the thickness
            else:
                (signal maybe too bad , ignore)
        else:
            (signal maybe too bad , ignore)
    """
    TOTAL_CHN, TOTAL_ROUND, SIGNAL_LENGTH = signal_matrices.shape
    for chn in range(TOTAL_CHN):        
        for rd in range(TOTAL_ROUND):
#            conv_result = find_envelope(np.convolve(signal_matrices[trLayout[chn], rd, :], S)) 
            conv_result = np.convolve(signal_matrices[trLayout[chn], rd, :], S) 
            peaks_locs = detect_peaks(conv_result, mph = 0.1 * conv_result.max(), mpd = 0.45 * TIME_FLIGHT, 
                                       edge = 'both', show=False)        
            peak_diff = np.diff(peaks_locs)
            if ( len(peak_diff) > 6 ):
                thickness_point = np.median(peak_diff[2:])
            elif (len(peak_diff) > 4):
                thickness_point = np.median(peak_diff[1:])
            elif (len(peak_diff) > 2):
                thickness_point = np.median(peak_diff[1:])
            elif (len(peak_diff) > 0):
                thickness_point = TIME_FLIGHT
            else:
                thickness_point = TIME_FLIGHT
            if (thickness_point <  TIME_FLIGHT * 1.1):
                    thickness_map[chn,rd] = thickness_point
            else:
                    thickness_map[chn,rd] = TIME_FLIGHT    
    return thickness_map

#-------------------------------------------------------------------------------
def OpenBinFile():
    name = filedialog.askopenfilename(initialdir = INITIAL_DIR,
                           filetypes =(("Binary File", "*.bin"),("All Files","*.*")),
                           title = "Choose a file.")
    print (name)
    #Using try in case user types in unknown file or closes without choosing a file.
    try:
        with open(name,'rb') as OpenedFile:
            SIGNAL_MATRICES, ROLL_R = processBinFile(OpenedFile)
            NORM_SIGNAL_MATRICES = take_3D_norm(SIGNAL_MATRICES)
#            TRIGGER_MAP.dump("data_trigger_map_" + name[-19:-4]  - ".bin")
#            SIGNAL_MATRICES.dump("data_NormSignal_matrices_" + name[-19:-4]  - ".bin")
#            roll_r.dump("roll_r_" + name[-19:-4] - ".bin")
            print("Done")
    except:
        print("No file exists")
        

def showdialog():
    '''各种窗口'''

    res = simpledialog.askstring(title='字符串', prompt='输入一个字符串')

    print(res)
    
    
class BlueNoseApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.geometry('1600x1000')                  # 窗口大小
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
        
        #for F in (StartPage, PageOne, PageThree):
        for F in (StartPage, SignalDetailPage ,GraphMainPage, CalliperPage, TimeDashBoard, EnergyPage):
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

        self.img0 = ImageTk.PhotoImage(Image.open('images/home.png'))
        self.img1 = ImageTk.PhotoImage(Image.open('images/binfile_ico.png'))
        self.img2 = ImageTk.PhotoImage(Image.open('images/caliper.png'))
        self.img3 = ImageTk.PhotoImage(Image.open('images/audio-wave.png'))
        self.img4 = ImageTk.PhotoImage(Image.open('images/save.png'))
        self.img5 = ImageTk.PhotoImage(Image.open('images/newspaper.png'))
        self.img6 = ImageTk.PhotoImage(Image.open('images/open-folder-with-document.png'))
        self.img7 = ImageTk.PhotoImage(Image.open('images/settings.png'))
        self.img8 = ImageTk.PhotoImage(Image.open('images/dashboard.png'))

    # 生成工具条
    def createToolbar(self):
        toolframe = tk.Frame(self, height=20, bg='#F7EED6')#, relief=tk.RAISED)
        frame = tk.Frame(toolframe, bg='#F7EED6')
        ttk.Button(frame, width=20, image=self.img1, command=OpenBinFile).grid(row=0, column=0, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img2, command=lambda: self.show_frame(CalliperPage)).grid(row=0, column=1, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img3, command=lambda: self.show_frame(TimeDashBoard)).grid(row=0, column=2, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img4, command=showdialog).grid(row=0, column=3, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img5, command=showdialog).grid(row=0, column=4, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img6, command=showdialog).grid(row=0, column=5, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img7, command=showdialog).grid(row=0, column=6, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img7, command=showdialog).grid(row=0, column=6, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img8, command=lambda: self.show_frame(EnergyPage)).grid(row=0, column=8, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=20, image=self.img0, command=lambda: self.show_frame(StartPage)).grid(row=0, column=7, padx=1, pady=1, sticky=tk.E)
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
        ttk.Button(self, text="Processing Folder", command = lambda: self.processing_folder(self.InputFolderPath.get())).grid(row = 3, column = 2, rowspan = 2, padx = 10 , pady = 10)

        # Multiple Bin File Analysis
        ttk.Button(self, text="Project Settings", command =showdialog).grid(row = 5, column = 1, padx = 10 , pady = 10)
        ttk.Button(self, text="Dashboard", command =showdialog).grid(row = 5, column = 2, padx = 10 , pady = 10)   
        ttk.Button(self, text="Tests", command =showdialog).grid(row = 5, column = 3, padx = 10 , pady = 10)        
        
        ttk.Button(self, text="Calliper & Movement",
                            command=lambda: controller.show_frame(CalliperPage)).grid(row = 6, column = 0, padx = 10 , pady = 10)
        
        ttk.Button(self, text="Signal & Details",
                            command=lambda: controller.show_frame(SignalDetailPage)).grid(row = 6, column =1, padx = 10 , pady = 10)
        
        ttk.Button(self, text="Thickness & Energy Analysis",
                            command=lambda: controller.show_frame(GraphMainPage)).grid(row = 6, column = 2, padx = 10 , pady = 10)
        
        ttk.Button(self, text="Dashboard",
                            command=lambda: controller.show_frame(TimeDashBoard)).grid(row = 6, column = 3, padx = 10 , pady = 10)
        
        
    def processing_folder(self, path):
        file_map = Dir_Parser(path)
        for minute_key, minute_file_list in file_map.items():
            file_num = len(minute_file_list)
            total_signal_matrices = np.empty((MATRICES_SIZE[0], MATRICES_SIZE[1] * file_num, MATRICES_SIZE[2]), dtype = 'float16')
            total_roll_r = np.zeros(( 260 * file_num , 1 ), dtype=np.uint16)
            index = 0
            for item in minute_file_list:
                with open(item, "rb") as bin_file:
                    total_signal_matrices[:, index*MATRICES_SIZE[1] : (index + 1)*MATRICES_SIZE[1], :], total_roll_r[index * 260 : (index + 1) * 260, :] = processBinFile(bin_file)
                    index = index + 1
            np.save(self.OutputFolderPath.get() + '/' + minute_key + '-signal', total_signal_matrices)
            np.save(self.OutputFolderPath.get() + '/' + minute_key + '-roll', total_roll_r)
        print("Finshied Stacking minute files")
        return
                    

class SettingInterface:
    """
    This interface make the settings in each Page coherent
    """
    def __init__(self):
        self.trLayout_sel_G = 0
        
    #def update(self):

setting_interface = SettingInterface()


class CalliperPage(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.initVariable(parent, controller, setting_interface)
        self.createPlots(parent, controller)   
        self.createWidgets(parent, controller) 
        
        self.bind("<Key>", self.key)
        self.bind("<Button-1>", self.click_callback)
        
        cid = self.canvas_calliper_plot.mpl_connect('button_press_event', self.onclick_caliper)   
        
    def initVariable(self, parent, controller, interface):
        self.trLayout = tk.StringVar()
        self.trLayout_sel = tk.IntVar()
        self.trLayout_sel.set(0)
        self.trLayout.set(  TR_LAYOUT_NAME[0] )
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
        tr_layout_box = ttk.Combobox(self, textvariable = self.trLayout, width = 35, values = TR_LAYOUT_NAME ) 
                                     #, postcommand = self.updtcblist(self, interface))
                                                                            
        tr_layout_box.grid(row = 2, column = 1, padx = 10 , pady = 10) #初始化  
        #trLayout_sel_G = tr_layout_box.current()
        
        
        tk.Label(self, text = "Channel").grid(row = 3, column = 0, padx = 10 , pady = 10)
        tk.Label(self, text = "Round").grid(row = 5, column = 0, padx = 10, pady = 10)
        tk.Entry( self, textvariable = self.channel_no ).grid(row=3, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.round_no ).grid(row=5, column=1, padx=10, pady=10)
        button_fw_chn = ttk.Button(self, text=">", command = lambda: self.chn_forward_callback(self.canvas_signal_plot, 
                                                                                         self.signal_plot_ax, 
                                                                                         self.canvas_offset_plot,
                                                                                         self.offset_plot_ax))
        button_fw_chn.grid(row = 4, column = 1, padx = 10 , pady = 10)
        button_fw_rd = ttk.Button(self, text=">", command = lambda: self.rd_forward_callback(self.canvas_signal_plot, 
                                                                                         self.signal_plot_ax, 
                                                                                         self.canvas_offset_plot,
                                                                                         self.offset_plot_ax))  

        button_fw_rd.grid(row = 6, column = 1, padx = 10 , pady = 10)        

        button_bw_chn = ttk.Button(self, text="<", command = lambda: self.chn_backward_callback(self.canvas_signal_plot, 
                                                                                         self.signal_plot_ax, 
                                                                                         self.canvas_offset_plot,
                                                                                         self.offset_plot_ax))  
        button_bw_chn.grid(row = 4, column = 0, padx = 10 , pady = 10)
        button_bw_rd = ttk.Button(self, text="<", command = lambda: self.rd_backward_callback(self.canvas_signal_plot, 
                                                                                         self.signal_plot_ax, 
                                                                                         self.canvas_offset_plot,
                                                                                         self.offset_plot_ax))  
        button_bw_rd.grid(row = 6, column = 0, padx = 10 , pady = 10)                
        
        tk.Label(self, text = "Start Delay", font = NORM_FONT).grid(row = 8, column = 0, padx = 10, pady = 10) 
        tk.Entry( self, textvariable = self.start_delay ).grid(row=8, column=1, padx=10, pady=10)
        ttk.Button(self, text="Signal Plot", command = lambda: self.signal_plot_callback(self.canvas_signal_plot, 
                                                                                         self.signal_plot_ax, 
                                                                                         self.canvas_offset_plot,
                                                                                         self.offset_plot_ax
                                                                                         )).grid(row = 9, column = 0, columnspan = 2, rowspan = 1, padx = 10, pady = 10)
        ttk.Button(self, text="Filtered Calliper Map Plot", command =lambda: self.calliper_plot_callback(self.canvas_calliper_plot, 
                                                                                                         self.calliper_plot_ax)).grid(row = 10, column = 0, columnspan = 2, rowspan = 1, padx = 10, pady = 10)
        ttk.Button(self, text="Save Map").grid(row = 13, column = 0, columnspan = 2, rowspan = 1, padx = 10, pady = 10)
        ttk.Button(self, text="Return To Home", 
                   command = lambda: controller.show_frame(StartPage)).grid(row = 14, column = 0, columnspan = 1, rowspan = 1, padx = 10, pady = 10)
        ttk.Button(self, text="Signal Channel Dashboard", 
                   command = lambda: controller.show_frame(TimeDashBoard)).grid(row = 14, column = 1, columnspan = 1, rowspan = 1, padx = 10, pady = 10)

    def key(self, event):
        """
        wasd,p
        """
        self.focus_set()
        if event.char == 'd':
            self.rd_forward_callback(self.canvas_signal_plot, 
                                self.signal_plot_ax, 
                                self.canvas_offset_plot,
                                self.offset_plot_ax)
        if event.char == 'a':
            self.rd_backward_callback(self.canvas_signal_plot, 
                                      self.signal_plot_ax, 
                                      self.canvas_offset_plot,
                                      self.offset_plot_ax)
        if event.char == 'w':
           self.chn_backward_callback(self.canvas_signal_plot, 
                                      self.signal_plot_ax, 
                                      self.canvas_offset_plot,
                                      self.offset_plot_ax) 
        if event.char == 's':
            self.chn_forward_callback(self.canvas_signal_plot, 
                                      self.signal_plot_ax, 
                                      self.canvas_offset_plot,
                                      self.offset_plot_ax)            
        if event.char == 'p':
            self.calliper_plot_callback(self.canvas_calliper_plot, 
                                        self.calliper_plot_ax)
                
    def click_callback(self, event):
        self.focus_set()
        print ("clicked at", event.x, event.y)
        
    def createPlots(self, parent, controller):
        signal_plot = Figure(figsize = (4,3))
        self.signal_plot_ax = signal_plot.add_subplot(111)        
        self.canvas_signal_plot = FigureCanvasTkAgg(signal_plot, self)
        self.canvas_signal_plot._tkcanvas.grid(row=2, column = 2, padx=0, pady=0 , columnspan = 3, rowspan = 4)
        self.canvas_signal_plot.show()     

        offset_plot = Figure(figsize = (4,3))
        self.offset_plot_ax = offset_plot.add_subplot(111)        
        self.canvas_offset_plot = FigureCanvasTkAgg(offset_plot, self)
        self.canvas_offset_plot._tkcanvas.grid(row=2, column = 5, padx=0, pady=0 , columnspan = 3, rowspan = 4)
        self.canvas_offset_plot.show()          
        
        calliper_plot = Figure(figsize = (8,4))
        self.calliper_plot_ax = calliper_plot.add_subplot(111)        
        self.canvas_calliper_plot = FigureCanvasTkAgg(calliper_plot, self)
        self.canvas_calliper_plot._tkcanvas.grid(row=6, column = 2, padx=0, pady=10 , columnspan = 8, rowspan = 4)
        self.canvas_calliper_plot.show()            

        
    def calliper_plot_callback(self,canvas,ax):
        Layout_index = TR_LAYOUT_NAME.index(self.trLayout.get())
        tr_layout = TR_LAYOUT[Layout_index]
        TRIGGER_MAP = calculate_trigger_map(NORM_SIGNAL_MATRICES, trLayout = tr_layout)
        TRIGGER_MAP = median_filter_2D(TRIGGER_MAP, 2)
        CALLIPER_MAP = calculate_calliper(TRIGGER_MAP)
        transducer_tdc = processRoll_r (ROLL_R)
        #
        ax.clear()
        ax.imshow(CALLIPER_MAP, aspect = 'auto',interpolation='none')
        ax.plot(transducer_tdc, 'r')
        canvas.draw()

    def signal_plot_callback(self, signal_canvas, signal_ax, position_canvas, position_ax):       
        c = ['r','b','g','k']  # plot marker colors
        rd, chn, start_delay, trLayout = self.get_info_from_self()
        signal_to_plot = NORM_SIGNAL_MATRICES[chn, rd, :]
        signal_ax.clear()
        position_ax.clear()
        signal_ax.plot(signal_to_plot, color=c[3])
        signal_ax.set_title('Full Signal Plot')
        theta = np.arange(0, 2*np.pi, 0.01)
        x = 0 + 0.9144 * np.cos(theta)
        y = 0 + 0.9144 * np.sin(theta)
        transducer_tdc = processRoll_r (ROLL_R)
        x_offset , y_offset = find_offset(transducer_tdc, chn, rd, CALLIPER_MAP)
        position_ax.plot(x, y)
        position_ax.plot(x_offset,y_offset,'ro',label="point")
        position_ax.axis('equal')
        position_ax.set_title('Offset Plot')
        position_ax.text(-1, 0.8, 'x offset =' +  "{0:.3f}%".format(x_offset*100) +';\n'+'y_offset = ' +  "{0:.3f}%".format(y_offset*100) , fontsize=10)
        
        signal_canvas.draw()
        position_canvas.draw()
        return
    
    
    def onclick_caliper(self, event):
        """
        onclick event for binding with canvas class
        """
        time_point = int(event.xdata)
        chn = int(event.ydata)  
        self.round_no.set(str(time_point))
        self.channel_no.set(str(chn))
        self.signal_plot_callback(self.canvas_signal_plot, self.signal_plot_ax, self.canvas_offset_plot,  self.offset_plot_ax)      
        
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, time_point, chn))        
            
    
    def get_info_from_self(self):
        rd = self.round_no.get()
        chn = self.channel_no.get()
        start_delay = self.start_delay.get()
        trLayout = list(self.trLayout.get())
        
        return rd, chn, start_delay, trLayout

    def chn_forward_callback(self, signal_canvas, signal_ax, position_canvas, position_ax):
        rd, chn, start_delay, trLayout = self.get_info_from_self()
        chn = (chn + 1) % MATRICES_SIZE[0]
        self.channel_no.set(chn)
        self.signal_plot_callback(signal_canvas, signal_ax, position_canvas, position_ax)
        
    def chn_backward_callback(self, signal_canvas, signal_ax, position_canvas, position_ax):
        rd, chn, start_delay, trLayout = self.get_info_from_self()
        chn = (chn - 1) % MATRICES_SIZE[0]
        self.channel_no.set(chn)
        self.signal_plot_callback(signal_canvas, signal_ax, position_canvas, position_ax)        
        
    def rd_forward_callback(self, signal_canvas, signal_ax, position_canvas, position_ax):
        rd, chn, start_delay, trLayout = self.get_info_from_self()
        rd = (rd + 1) % MATRICES_SIZE[1]
        self.round_no.set(rd)
        self.signal_plot_callback(signal_canvas, signal_ax, position_canvas, position_ax)
        
    def rd_backward_callback(self, signal_canvas, signal_ax, position_canvas, position_ax):
        rd, chn, start_delay, trLayout = self.get_info_from_self()
        rd = (rd - 1) % MATRICES_SIZE[1]
        self.round_no.set(rd)
        self.signal_plot_callback(signal_canvas, signal_ax, position_canvas, position_ax)      




class SignalDetailPage(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.chn_no = tk.StringVar() 
        self.time_point = tk.StringVar() 
        self.mph = tk.StringVar() 
        self.mpd = tk.StringVar() 
        self.chn_no.set("0")
        self.time_point.set("0")        
        self.mph.set("0.1")
        self.mpd.set("25")
        self.createWidgets(parent, controller)

    
    def createWidgets(self, parent, controller):
        # First ROW      
        tk.Label(self, text="Signal & Detailed Time Analysis", font=LARGE_FONT).grid(row=0, column = 2, columnspan = 2, rowspan = 2,  padx=10, pady=10)
        ttk.Separator(self, orient= 'horizontal').grid(row = 2, column = 0, columnspan = 6, sticky="ew",  padx=10, pady=10)

        tk.Label(self, text="Channel", font=NORM_FONT).grid(row=3, column=0, padx=10, pady=10, sticky='e')
        tk.Label(self, text="Time Point", font=NORM_FONT).grid(row=4, column=0, padx=10, pady=10, sticky='e')
        tk.Entry( self, textvariable = self.chn_no ).grid(row=3, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.time_point ).grid(row=4, column=1, padx=10, pady=10)

        tk.Label(self, text="mph", font=NORM_FONT).grid(row=5, column=0, padx=10, pady=10, sticky='e')
        tk.Label(self, text="mpd", font=NORM_FONT).grid(row=6, column=0, padx=10, pady=10, sticky='e')
        tk.Entry( self, textvariable = self.mph ).grid(row=5, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.mpd ).grid(row=6, column=1, padx=10, pady=10)

        
        ttk.Button(self, text="Show",
                            command=lambda: self.plot_callback(canvas_plot1, signal_plot1, envelope_plot1, correlation_plot1, correlation_plot1_peak)).grid(row=7, column = 0 , columnspan = 2,  padx=10, pady=10)
        
        #--------------------Canvas for Plotting-------------------------
        canvas_signal1 = Figure() # Signal Detail
        signal_plot1 = canvas_signal1.add_subplot(221)
        envelope_plot1 =  canvas_signal1.add_subplot(223)
        correlation_plot1 =  canvas_signal1.add_subplot(222)
        correlation_plot1_peak = canvas_signal1.add_subplot(224)
        canvas_plot1 = FigureCanvasTkAgg(canvas_signal1, self)
        canvas_plot1._tkcanvas.grid(row=3, column = 3, padx=10, pady=10 , columnspan = 4, rowspan = 5)
        canvas_plot1.show()
        #--------------------BOTTOM BAR---------------------------
        ttk.Separator(self, orient= 'horizontal').grid(row = 8, column = 0, columnspan = 6, sticky="ew",  padx=10, pady=10)
        ttk.Button(self, text="<-- Back to Home",
                            command=lambda: controller.show_frame(StartPage)).grid(row=9, column = 0 , columnspan = 1,  padx=10, pady=10)        
        ttk.Button(self, text="<-- Calliper",
                            command=lambda: controller.show_frame(CalliperPage)).grid(row=9, column = 1 , columnspan = 1,  padx=10, pady=10)    
        ttk.Button(self, text="Thickness-->",
                            command=lambda: controller.show_frame(CalliperPage)).grid(row=9, column = 2 , columnspan = 1,  padx=10, pady=10)

    def plot_callback(self,canvas,ax1, ax2, ax3, ax4):
        c = ['r','b','g']  # plot marker colors
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        rd = int(self.time_point.get())
        chn = int(self.chn_no.get())
        mph = float(self.mph.get())
        mpd = int(self.mpd.get())
        signal_plot = NORM_SIGNAL_MATRICES[chn, rd, :]
        ax1.plot(signal_plot, color=c[0])
        detect_peaks(signal_plot, mph=mph, mpd=mpd, edge='rising', show=True, ax=ax2)
        ax3.plot(np.convolve(signal_plot, S), color=c[0])
        detect_peaks(np.convolve(signal_plot, S), mph=mph, mpd=mpd, edge='rising', show=True, ax=ax4)   
        canvas.draw()
        

        

           
class GraphMainPage(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.createWidgets(parent, controller)
        
    def createWidgets(self, parent, controller):    
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
#        cid = canvas.mpl_connect('button_press_event', onclick)
        
        #--------------------BUtton control refreshing-------------------------     
        button_thickness = ttk.Button(self, text="UpdateThickness",
                            command=lambda: self.thickness_update(canvas, thinkness_plot))
        button_thickness.pack()       
        
        button2 = ttk.Button(self, text="Refresh",
                            command=lambda: self.plot_callback(canvas, signal_plot, time_energy_plot))
        button2.pack()            
        
    def plot_callback(self,canvas,ax1, ax2):
        c = ['r','b','g']  # plot marker colors
        ax1.clear()
        ax2.clear()
        signal_plot = NORM_SIGNAL_MATRICES[0, 0, :]
        shape = (NORM_SIGNAL_MATRICES[0, 250:1000]).astype(float)
        ax1.plot(signal_plot, color=c[0])
        ax2.imshow(shape.transpose(), aspect = 'auto',interpolation='none')
        canvas.draw()
        
    def thickness_update(self, canvas, ax):
        ax.clear()
        thickness_map = Thickness_map_TIME(NORM_SIGNAL_MATRICES)
        ax.imshow(thickness_map, aspect = 'auto',interpolation='none')
        canvas.draw()        

class TimeDashBoard(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        self.initEntry()
        self.initLabelString()
        self.createWidgets(parent, controller)
        
        
    def initEntry(self):
        self.chn_no = tk.StringVar()
        self.time_point = tk.StringVar() 
        self.mph = tk.StringVar() 
        self.mpd = tk.StringVar() 
        self.chn_no.set("0")
        self.time_point.set("0")        
        self.mph.set("0.1")
        self.mpd.set("25")     
        self.low_margin = tk.StringVar() 
        self.high_margin = tk.StringVar() 
        self.low_margin.set("0")
        self.high_margin.set("1000") 

        
    def initLabelString(self):
        self.df_original = tk.StringVar()
        self.df_correlation = tk.StringVar()
        self.df_hilbert = tk.StringVar()
        self.median_original = tk.StringVar()
        self.median_correlation = tk.StringVar()
        self.median_hilbert = tk.StringVar()
        self.mean_original = tk.StringVar()
        self.mean_correlation = tk.StringVar()
        self.mean_hilbert = tk.StringVar()        
        self.df_original.set("DF: NA")
        self.df_correlation.set("DF: NA")
        self.df_hilbert.set("DF: NA")
        self.median_original.set("MEDIAN: NA")
        self.median_correlation.set("MEDIAN: NA")
        self.median_hilbert.set("MEDIAN: NA")
        self.mean_original.set("MEAN: NA")
        self.mean_correlation.set("MEAN: NA")
        self.mean_hilbert.set("MEAN: NA")           
        
        
        
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
        tk.Entry( self, textvariable = self.chn_no ).grid(row=2, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.time_point ).grid(row=4, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.mph ).grid(row=6, column=1, padx=10, pady=10)
        tk.Entry( self, textvariable = self.mpd ).grid(row=7, column=1, padx=10, pady=10)
        
        
        tk.Label(self, text="Map Margin Setting", 
                 font=NORM_FONT).grid(row=8, column = 0, columnspan = 2, padx=10, pady=10)  
        tk.Entry( self, textvariable = self.low_margin ).grid(row=9, column=0, padx=10, pady=10)
        tk.Entry( self, textvariable = self.high_margin ).grid(row=9, column=1, padx=10, pady=10)   
        
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
         
        refresh_signal_plot = ttk.Button(self, text="Refresh Signal", 
                                         command=lambda: self.refresh_signal_callback(canvas_full_signal_plot, full_signal_plot_ax, 
                                                                            canvas_fp_partial, fp_partial_ax, 
                                                                            canvas_fp_corrlation, fp_corrlation_ax,
                                                                            canvas_fp_hilbert, fp_hilbert_ax))
        refresh_signal_plot.grid(row=11, column = 0 , columnspan = 1,  padx=10, pady=10)     

        refresh_channel_plot = ttk.Button(self, text="Refresh Channel", command=lambda: self.refresh_channel_callback(canvas_full_signal_plot, full_signal_plot_ax, 
                                                                            canvas_fp_partial, fp_partial_ax, 
                                                                            canvas_fp_corrlation, fp_corrlation_ax,
                                                                            canvas_fp_hilbert, fp_hilbert_ax,
                                                                            canvas_signel_transducer_plot, signel_transducer_plot_ax))
        refresh_channel_plot.grid(row=11, column = 1 , columnspan = 1,  padx=10, pady=10)            
        
        backward_round_botton = ttk.Button(self, text="<", command=lambda: self.backward_round(canvas_full_signal_plot, full_signal_plot_ax, 
                                                                            canvas_fp_partial, fp_partial_ax, 
                                                                            canvas_fp_corrlation, fp_corrlation_ax,
                                                                            canvas_fp_hilbert, fp_hilbert_ax))
        backward_round_botton.grid(row=5, column = 0 , columnspan = 1,  padx=10, pady=10)    

        forward_round_botton = ttk.Button(self, text=">", command=lambda:  self.forward_round(canvas_full_signal_plot, full_signal_plot_ax, 
                                                                            canvas_fp_partial, fp_partial_ax, 
                                                                            canvas_fp_corrlation, fp_corrlation_ax,
                                                                            canvas_fp_hilbert, fp_hilbert_ax))
        forward_round_botton.grid(row=5, column = 1 , columnspan = 1,  padx=10, pady=10)   
         

        backward_chn_botton = ttk.Button(self, text="<", command=lambda: self.backward_chn(canvas_full_signal_plot, full_signal_plot_ax, 
                                                                            canvas_fp_partial, fp_partial_ax, 
                                                                            canvas_fp_corrlation, fp_corrlation_ax,
                                                                            canvas_fp_hilbert, fp_hilbert_ax,
                                                                            canvas_signel_transducer_plot, signel_transducer_plot_ax))
        backward_chn_botton.grid(row=3, column = 0 , columnspan = 1,  padx=10, pady=10)    

        forward_chn_button = ttk.Button(self, text=">", command=lambda:  self.forward_chn(canvas_full_signal_plot, full_signal_plot_ax, 
                                                                            canvas_fp_partial, fp_partial_ax, 
                                                                            canvas_fp_corrlation, fp_corrlation_ax,
                                                                            canvas_fp_hilbert, fp_hilbert_ax,
                                                                            canvas_signel_transducer_plot, signel_transducer_plot_ax))
        forward_chn_button.grid(row=3, column = 1 , columnspan = 1,  padx=10, pady=10) 
        ## THICKNESS BUTTONs
        update_thickness_1 = ttk.Button(self, text="Update Thickness 1", command=lambda: self.thickness_update(canvas_thickness_plot, thickness_plot_ax))
        update_thickness_1.grid(row=13, column = 0 , columnspan = 2,  padx=10, pady=10)  
        update_thickness_2 = ttk.Button(self, text="Update Thickness 2", command=lambda: controller.show_frame(CalliperPage))
        update_thickness_2.grid(row=14, column = 0 , columnspan = 2,  padx=10, pady=10)          

        revise_thickness = ttk.Button(self, text="Revise Thickness", command=lambda: controller.show_frame(CalliperPage))
        revise_thickness.grid(row=16, column = 0 , columnspan = 2,  padx=10, pady=10)           
        
        cid = canvas_thickness_plot.mpl_connect('button_press_event', self.onclick)        
        
    def refresh_signal_callback(self,canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4):
        c = ['r','b','g','k']  # plot marker colors
        rd, chn, mph, mpd = self.get_info_from_self()
        signal_to_plot = NORM_SIGNAL_MATRICES[chn, rd, :]
        first_index_to_come = min(np.argmax(signal_to_plot), np.argmin(signal_to_plot))
        if (first_index_to_come > 200) and (first_index_to_come < 1400):
            signal_slice_to_plot = signal_to_plot[first_index_to_come - 200 : first_index_to_come + 600]
        elif (first_index_to_come < 200):
            signal_slice_to_plot = signal_to_plot[0 : 800]
        else:
            signal_slice_to_plot = signal_to_plot[-801:]
        
        xcorr_signal = np.convolve(signal_slice_to_plot, S)
        amplitude_envelope = find_envelope(xcorr_signal)
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        ax1.plot(signal_to_plot, color=c[3])
        peak_pos = detect_peaks(signal_slice_to_plot, mph=mph, mpd=mpd, edge='rising', show=True, ax=ax2)
        peak_diff = np.diff(peak_pos)
        self.df_original.set("DF "+str(peak_diff))
        if len(peak_diff) > 0:
            median = np.median(peak_diff)
            mean = np.median(peak_diff)
            self.median_original.set("MEDIAN "+ str(median))
            self.mean_original.set("MEAN "+ str(mean))
        peak_pos = detect_peaks(xcorr_signal  , mph=mph*xcorr_signal.max(), mpd=mpd, edge='rising', show=True, ax=ax3)
        peak_diff = np.diff(peak_pos)
        self.df_correlation.set(str(peak_diff))
        if len(peak_diff) > 0:
            median = np.median(peak_diff)
            mean = np.median(peak_diff)
            self.median_correlation.set("MEDIAN "+ str(median))
            self.mean_correlation.set("MEAN "+ str(mean))        
        detect_peaks(amplitude_envelope, mph=mph*amplitude_envelope.max(), mpd=mpd, edge='rising', show=True, ax=ax4)
        peak_diff = np.diff(peak_pos)        
        self.df_hilbert.set(str(peak_diff))        
        if len(peak_diff) > 0:
            median = np.median(peak_diff)
            mean = np.median(peak_diff)
            self.median_hilbert.set("MEDIAN "+ str(median))
            self.mean_hilbert.set("MEAN "+ str(mean)) 
        canvas1.draw()
        canvas2.draw()
        canvas3.draw()
        canvas4.draw() 
    
    def refresh_channel_callback(self, canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4, canvas5, ax5):
        rd, chn, mph, mpd = self.get_info_from_self()
        self.refresh_signal_callback(canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4)
        
        low_margin = int(self.low_margin.get())
        high_margin = int(self.high_margin.get())
        ax5.clear()
        shape = (NORM_SIGNAL_MATRICES[chn, :, low_margin:high_margin]).astype(float)
        ax5.imshow(shape.transpose(), aspect = 'auto',interpolation='none', cmap = plt.cm.jet)
        ax5.vlines(rd, low_margin, high_margin, colors = "c", linestyles = "dashed")
        canvas5.draw()         
        
    def forward_round(self,canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4):
        rd, chn, mph, mpd = self.get_info_from_self()
        rd = (rd + 1) % MATRICES_SIZE[1]
        self.time_point.set(str(rd))
        self.refresh_signal_callback(canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4)
        
    def backward_round(self,canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4):    
        rd, chn, mph, mpd = self.get_info_from_self()
        rd = (rd - 1) % MATRICES_SIZE[1]
        self.time_point.set(str(rd))
        self.refresh_signal_callback(canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4)
        
        
    def forward_chn(self,canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4, canvas5, ax5):
        rd, chn, mph, mpd = self.get_info_from_self()
        chn = (chn + 1) % MATRICES_SIZE[0]
        self.chn_no.set(str(chn))
        self.refresh_channel_callback(canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4, canvas5, ax5)
        
    def backward_chn(self,canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4, canvas5, ax5):    
        rd, chn, mph, mpd = self.get_info_from_self()
        chn = (chn - 1) % MATRICES_SIZE[0]
        self.chn_no.set(str(chn))
        self.refresh_channel_callback(canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4, canvas5, ax5)        
        
    def get_info_from_self(self):
        rd = int(self.time_point.get())
        chn = int(self.chn_no.get())
        mph = float(self.mph.get())
        mpd = int(self.mpd.get())
        return rd, chn, mph, mpd
  
    def thickness_update(self, canvas, ax):
        ax.clear()
        thickness_map = Thickness_map_TIME(NORM_SIGNAL_MATRICES)
        ax.imshow(thickness_map, aspect = 'auto',interpolation='none')
        canvas.draw()        

    def onclick(self, event):
        """
        onclick event for binding with canvas class
        """
        time_point = int(event.xdata)
        chn = int(event.ydata)  
        self.time_point.set(str(time_point))
        self.chn_no.set(str(chn))
#        self.refresh_signal_callback(canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4)
        
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, time_point, chn))        


class EnergyPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.initVariable(parent, controller)
        self.createWidgets(parent, controller)
        self.set_control_panel(parent, controller)
        
    def initVariable(self, parent, controller):
        self.trLayout = tk.StringVar()
        self.trLayout_sel = tk.IntVar()
        self.trLayout_sel.set(0)
        self.trLayout.set(  TR_LAYOUT_NAME[0] )        
        self.channel_no = tk.IntVar()
        self.channel_no.set(0)
        self.range_low = tk.IntVar()
        self.range_high = tk.IntVar()
        self.range_low.set(0)
        self.range_high.set(500)
        self.metal_v = tk.IntVar()
        self.metal_v.set(5920)
        self.coating_v = tk.IntVar()
        self.coating_v.set(3400)
#        layout of the transducers
#        focus channels
#
#        metal velocity
#
#        coating velocity
    def createWidgets(self, parent, controller):
        return
    
    def set_control_panel(self, parent, controller):
        toolframe = tk.Frame(self, height=15, bg='#F7EED6')#, relief=tk.RAISED)
        frame = tk.Frame(toolframe, bg='#F7EED6')
                         
                         
        tk.Label(frame, text="TrLayout", 
                 font=NORM_FONT).grid(row=0, column=0, padx=1, pady=1, sticky=tk.E)
        tr_layout_box = ttk.Combobox(frame, textvariable = self.trLayout, width = 10, values = TR_LAYOUT_NAME )
        
        tr_layout_box.grid(row=0, column=1, padx=1, pady=1, sticky=tk.E)
        
        tk.Label(frame, text= "Metal Velocity (m/s)", font=NORM_FONT).grid(row=0, column=2, padx=1, pady=1, sticky=tk.E)
        tk.Entry( frame, textvariable = self.metal_v ).grid(row=0, column=3, padx=1, pady=1, sticky=tk.E)
        tk.Label(frame, text= "Coating Velocity (m/s)", font=NORM_FONT).grid(row=0, column=4, padx=1, pady=1, sticky=tk.E)
        tk.Entry( frame, textvariable = self.coating_v ).grid(row=0, column=5, padx=1, pady=1, sticky=tk.E)
        tk.Label(frame, text= "Focus Channel", font=NORM_FONT).grid(row=0, column=6, padx=1, pady=1, sticky=tk.E)
        tk.Entry( frame, textvariable = self.channel_no ).grid(row=0, column=7, padx=1, pady=1, sticky=tk.E)
        tk.Label(frame, text= "Time Point Range", font=NORM_FONT).grid(row=0, column=8, padx=1, pady=1, sticky=tk.E)
        tk.Entry( frame, textvariable = self.range_low ).grid(row=0, column=9, padx=1, pady=1, sticky=tk.E)
        tk.Entry( frame, textvariable = self.range_high ).grid(row=0, column=10, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=15,text= "Plot Heat Map",  command=showdialog).grid(row=0, column=11, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=15, text= "Plot Energy Map",  command=showdialog).grid(row=0, column=12, padx=1, pady=1, sticky=tk.E)
        ttk.Button(frame, width=15, text= "Plot All",  command=showdialog).grid(row=0, column=13, padx=1, pady=1, sticky=tk.E)
        frame.pack(side=tk.LEFT)
        toolframe.pack(fill=tk.X)
        
        map_frame = tk.Frame(self, height=605, width = 500, bg='#000000')
        heat_map_frame = tk.Frame(map_frame, bg='#000000')
        energy_map_frame = tk.Frame(map_frame, bg = "#000000")                          
        
        channel_m2_plot = Figure(figsize = (4,2.5))
        channel_m2_plot_ax =  channel_m2_plot.add_subplot(111)
        channel_m2_plot_canvas = FigureCanvasTkAgg(channel_m2_plot, heat_map_frame)
        channel_m2_plot_canvas._tkcanvas.grid(row=0, column = 0, padx=10, pady=10 , columnspan = 4, rowspan = 3)
        channel_m2_plot_canvas.show()        

        channel_m1_plot = Figure(figsize = (4,2.5))
        channel_m1_plot_ax =  channel_m1_plot.add_subplot(111)
        channel_m1_plot_canvas = FigureCanvasTkAgg(channel_m1_plot, heat_map_frame)
        channel_m1_plot_canvas._tkcanvas.grid(row=0, column = 4, padx=10, pady=10 , columnspan = 4, rowspan = 3)
        channel_m1_plot_canvas.show()
#        
        channel_m0_plot = Figure(figsize = (4,2.5))
        channel_m0_plot_ax =  channel_m0_plot.add_subplot(111)
        channel_m0_plot_canvas = FigureCanvasTkAgg(channel_m0_plot, heat_map_frame)
        channel_m0_plot_canvas._tkcanvas.grid(row=3, column = 0, padx=10, pady=10 , columnspan = 4, rowspan = 3)
        channel_m0_plot_canvas.show()        
##
        channel_a1_plot = Figure(figsize = (4,2.5))
        channel_a1_plot_ax =  channel_a1_plot.add_subplot(111)
        channel_a1_plot_canvas = FigureCanvasTkAgg(channel_a1_plot, heat_map_frame)
        channel_a1_plot_canvas._tkcanvas.grid(row=3, column = 4, padx=10, pady=10 , columnspan = 4, rowspan = 3)
        channel_a1_plot_canvas.show()        
        
        channel_a2_plot = Figure(figsize = (4,2.5))
        channel_a2_plot_ax =  channel_a2_plot.add_subplot(111)
        channel_a2_plot_canvas = FigureCanvasTkAgg(channel_a2_plot, heat_map_frame)
        channel_a2_plot_canvas._tkcanvas.grid(row=6, column = 0, padx=10, pady=10 , columnspan = 4, rowspan = 3)
        channel_a2_plot_canvas.show()    
##
        channel_a3_plot = Figure(figsize = (4,2.5))
        channel_a3_plot_ax =  channel_a3_plot.add_subplot(111)
        channel_a3_plot_canvas = FigureCanvasTkAgg(channel_a3_plot, heat_map_frame)
        channel_a3_plot_canvas._tkcanvas.grid(row=6, column = 4, padx=10, pady=10 , columnspan = 4, rowspan = 3)
        channel_a3_plot_canvas.show()           
#        
        energy_map_plot = Figure(figsize = (4,8))
        energy_map_plot_ax =  energy_map_plot.add_subplot(111)
        energy_map_plot_canvas = FigureCanvasTkAgg(energy_map_plot, energy_map_frame)
        energy_map_plot_canvas._tkcanvas.grid(row=0, column = 0, padx=10, pady=10 , columnspan = 4, rowspan = 3)
        energy_map_plot_canvas.show()          
        
        
        energy_map_plot1 = Figure(figsize = (4,8))
        energy_map_plot1_ax =  energy_map_plot1.add_subplot(111)
        energy_map_plot1_canvas = FigureCanvasTkAgg(energy_map_plot1, energy_map_frame)
        energy_map_plot1_canvas._tkcanvas.grid(row=0, column = 4, padx=10, pady=10 , columnspan = 4, rowspan = 3)
        energy_map_plot1_canvas.show()           
        
        
        heat_map_frame.pack(side=tk.LEFT)
        energy_map_frame.pack(side = tk.RIGHT)
        map_frame.pack(fill = tk.Y)      
        
                              
#    	control_frame = tk.Frame(self, ...) # toolframe = tk.Frame(self, height=20, bg='#F7EED6')#, relief=tk.RAISED)
#
#    	button
#
#    	entry
#
#    	control_frame.pack(side = tk.TOP)


    def set_plot_panel(self, parent, controller):
    	return
    
    
    def plot_heat_map(self, channel, tp_low, tp_high, canvas1, ax1, canvas2, ax2, canvas3, ax3, canvas4, ax4, canvas5, ax5, canvas6, ax6):
        return
    
    def get_info_from_self(self):
        chn = self.channel_no.get()
     
app = BlueNoseApp()
#ani = animation.FuncAnimation(f, animate, interval=10000)
app.mainloop()    
