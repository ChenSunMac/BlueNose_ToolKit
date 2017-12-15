# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:41:49 2017

@author: Chens
"""
import numpy as np
import time
import os

start_TOTAL = time.time()
file_name = "bn170614-142902.bin"
test_path = "C:\\Users\\chens\\Documents\\gui-dev\\SmallTempData\\0.3_ft_Run2"
if os.path.exists(test_path):
    os.chdir(test_path)
currentPath = os.getcwd()
print(currentPath)


start_PROCESSING = time.time()
bin_file_size = os.stat(file_name).st_size

with open(file_name, "rb") as bin_file:
    raw_data = np.fromfile(bin_file, dtype = np.uint8)
    bin_file_size = len(raw_data)
MATRICES_SIZE = (96, 520, 2000)
ii = np.zeros((1,128), dtype=np.int)
start_byte = 0
rp_i = 0
rp_locs = np.zeros(6240)
signal_matrices = np.zeros(MATRICES_SIZE)
for i in range(1, int(bin_file_size/32096) + 1):
    raw_fire_time = raw_data[start_byte + 24:start_byte + 32]
    roll_b = raw_data[start_byte + 16:start_byte + 18].view('int16')
    pitch_b = raw_data[start_byte + 18:start_byte + 20].view('int16')
    if((roll_b != 8224) | (pitch_b != 8224)):
        rp_locs[rp_i] = i
        rp_i = rp_i + 1
        
    for k in range(0, 8):
        raw_signal = raw_data[start_byte + k * 4008 + 40 : start_byte + k * 4008 + 4040].view('uint16')
        raw_signal = (raw_signal.astype("double")-32768)/32768
        raw_signal = np.asmatrix(raw_signal)
        #raw_first_ref = raw_data[start_byte+k*4008+32:start_byte +k*4008+34]
        #first_ref = raw_first_ref.view('uint16')
        channel_index = raw_data[start_byte + k*4008 + 38].astype("int64")
        # FUTURE : add thickness and distance calculation here to save more time
        signal_matrices[channel_index, ii[0,channel_index], :] = raw_signal
        ii[0,channel_index] = ii[0,channel_index] + 1
    start_byte = start_byte +32096
