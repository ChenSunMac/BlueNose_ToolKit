# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:45:30 2018

@author: Chens
"""
#---------------------
import os
test_path = "C:\\Users\\chens\\Desktop\\1.3fts"
if os.path.exists(test_path):
    os.chdir(test_path)

import glob 
import numpy as np 

import matplotlib.pyplot as plt # for plot
import seaborn as sns # for nice plot
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
   
#------------- Constant Configure -----------------------------
trLayout = [1, 33, 17, 29, 13, 93, 49, 81, 65, 77, 61, 21, 25, 9, 41, 5, 37,
            69, 73, 57, 89, 53, 85, 45, 2, 34, 18, 30, 14, 94, 50, 82, 66, 78,
            62, 22, 26, 10, 42, 6, 38, 70, 74, 58, 90, 54, 86, 46, 3, 35, 19, 
            31, 15, 95, 51, 83, 67, 79, 63, 23, 27, 11, 43, 7, 39, 71, 75, 59,
            91, 55, 87, 47, 4, 36, 20, 32, 16, 96, 52, 84, 68, 80, 64, 24, 28,
            12, 44, 8, 40, 72, 76, 60, 92, 56, 88, 48]

#trLayout = np.linspace(1, 96, 96, dtype = 'uint')
trLayout =  np.asarray(trLayout) - 1
MATRICES_SIZE = (96, 520, 2000)

START_DELAY = 6601
#------------- USEFUL FUNCTIONS -----------------------------
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
                
    return file_map

def processBinFile(OpenedFile):
    """
    with an opend file, 
    """
    raw_data = np.fromfile(OpenedFile, dtype = np.uint8)
    bin_file_size = len(raw_data)  
    ii = np.zeros((1,128), dtype=np.int)
    start_byte = 0
    rp_i = 0
    rp_locs = np.zeros(6240, dtype='int')   
    signal_matrices = np.empty(MATRICES_SIZE, dtype = 'float16')
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
            signal_matrices[channel_index, ii[0,channel_index], :] = raw_signal
            ii[0,channel_index] = ii[0,channel_index] + 1
        start_byte = start_byte +32096
    return signal_matrices


file_map = Dir_Parser(test_path)

for minute_key, minute_file_list in file_map.items():
    # do something for the key if multiple keys
    file_num = len(minute_file_list)
    total_signal_matrices = np.empty((MATRICES_SIZE[0], MATRICES_SIZE[1] * file_num, MATRICES_SIZE[2]), dtype = 'float16')
    index = 0
    for item in minute_file_list:
        with open(item, "rb") as bin_file:
            total_signal_matrices[:, index*MATRICES_SIZE[1] : (index + 1)*MATRICES_SIZE[1], :] = processBinFile(bin_file)
            index +=  1    

# ------------------------- Processing the HUGE signal_map

def trigger_map(signal_matrices):
    TOTAL_CHN, TOTAL_ROUND, SIGNAL_LENGTH = signal_matrices.shape
    trigger_map = np.zeros((TOTAL_CHN, TOTAL_ROUND),  dtype='uint16')
    for chn in range(TOTAL_CHN):
        for rd in range(TOTAL_ROUND):
            signal = signal_matrices[trLayout[chn], rd, :]
            norm_signal = np.float16( signal/np.max(np.absolute(signal)))
            # USE Numpy.argmax instead of for loop to save time
            trigger = np.argmax((np.absolute(norm_signal) > 0.594)) # use Absolute in case the negative edge coming first (Phase)
            if (trigger < 20) or (trigger > 1700):
                trigger = 20
            else:
                pass
            trigger_map[chn, rd] = trigger
    return trigger_map
trigger_map = trigger_map(total_signal_matrices)

# Flatten the boundary of the map:
FILTER_DIM = 2 # corresponding kernel size is 2*FILTER_DIM + 1 = 5

top_row = trigger_map[0 : FILTER_DIM, :]
bot_row = trigger_map[-FILTER_DIM : , :]

var_map1 =  np.vstack((trigger_map,top_row))
var_map2 =  np.vstack((bot_row,var_map1))

left_col = var_map2[:, 0 : FILTER_DIM]
right_col = var_map2[:, -FILTER_DIM : ]

var_map3 =  np.hstack((var_map2,right_col))
var_map4 =  np.hstack((left_col,var_map3))
# Median Filter
import scipy.signal
var_map5 = scipy.signal.medfilt(var_map4)

final_map = np.int16(var_map5[FILTER_DIM: -FILTER_DIM, FILTER_DIM:-FILTER_DIM])

plt.subplot(2, 2, 1)
sns.heatmap(trigger_map, vmin = 0, vmax = 1700, cbar=False)
plt.subplot(2, 2, 2)
sns.heatmap(var_map5, vmin = 0, vmax = 1700, cbar=False)
plt.subplot(2, 2, 3)
sns.heatmap(final_map, vmin = 0, vmax = 1700, cbar=False)
plt.subplot(2, 2, 4)
sns.heatmap(np.absolute(trigger_map - final_map), vmin = 0, vmax = 600, cbar=False)
plt.show()

def calculate_calliper(signal_matrices, trigger_map):
    TOTAL_CHN, TOTAL_ROUND, SIGNAL_LENGTH = signal_matrices.shape
    main_signal_matrices = np.zeros((TOTAL_CHN, TOTAL_ROUND, 300),  dtype='float16')
    calliper_map = np.zeros((TOTAL_CHN, TOTAL_ROUND),  dtype='float16')
    for chn in range(TOTAL_CHN):
        for rd in range(TOTAL_ROUND):
            calliper_map[chn,rd] = np.float16( (START_DELAY + trigger_map[chn,rd])*740.0/15000000)
            signal = signal_matrices[trLayout[chn], rd, :]
            norm_signal = np.float16( signal/np.max(np.absolute(signal)))
            main_reflection = norm_signal[trigger_map[chn,rd] - 20 : trigger_map[chn,rd] + 280]
            main_signal_matrices[chn, rd, :] = main_reflection
    return main_signal_matrices, calliper_map

# main_signal_matrices is normalized signal matrix
main_signal_matrices, calliper_map = calculate_calliper(total_signal_matrices, final_map)

plt.subplot(2, 2, 1)
sns.heatmap(np.absolute(np.transpose(main_signal_matrices[47, :, :])))
plt.subplot(2, 2, 2)
sns.heatmap(np.absolute(np.transpose(main_signal_matrices[55, :, :])))
plt.subplot(2, 2, 3)
sns.heatmap(np.absolute(np.transpose(main_signal_matrices[57, :, :])))
plt.subplot(2, 2, 4)
sns.heatmap(np.absolute(np.transpose(main_signal_matrices[85, :, :])))
plt.show()          
            

            