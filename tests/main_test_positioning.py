# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:41:49 2017

@author: Chens
"""
import numpy as np
import time
import os
import pickle

import pywt

#------------- Constant Configure -----------------------------
trLayout = [1, 33, 17, 29, 13, 93, 49, 81, 65, 77, 61, 21, 25, 9, 41, 5, 37,
            69, 73, 57, 89, 53, 85, 45, 2, 34, 18, 30, 14, 94, 50, 82, 66, 78,
            62, 22, 26, 10, 42, 6, 38, 70, 74, 58, 90, 54, 86, 46, 3, 35, 19, 
            31, 15, 95, 51, 83, 67, 79, 63, 23, 27, 11, 43, 7, 39, 71, 75, 59,
            91, 55, 87, 47, 4, 36, 20, 32, 16, 96, 52, 84, 68, 80, 64, 24, 28,
            12, 44, 8, 40, 72, 76, 60, 92, 56, 88, 48]

#trLayout = np.linspace(1, 96, 96, dtype = 'uint')
trLayout =  np.asarray(trLayout) - 1

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
rp_locs = np.zeros(6240, dtype='int')
signal_matrices = np.zeros(MATRICES_SIZE,  dtype='float16')
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

print("Total Time is about ", time.time() - start_PROCESSING, "to read binary file")

#-------------------------------Calliper---------------------------------
MATRIX_SIZE = (96, 520)
distance = np.zeros(MATRIX_SIZE,  dtype='int32')
main_signal_matrices = np.zeros((96, 520 , 400),  dtype='float16')
START_DELAY = 6601
TOTAL_CHN, TOTAL_ROUND, SIGNAL_LENGTH = signal_matrices.shape

for chn in range(TOTAL_CHN):
    for rd in range(TOTAL_ROUND):
        signal = signal_matrices[trLayout[chn], rd, :]
        norm_signal = np.float16( signal/np.max(np.absolute(signal)))
        # USE Numpy.argmax instead of for loop to save time
        trigger = np.argmax((np.absolute(norm_signal) > 0.594))
        if (trigger < 20) or (trigger > 1700):
            trigger = 20
        else:
            pass
        main_reflection = norm_signal[trigger - 20 : trigger + 380]
        main_signal_matrices[chn, rd, :] = main_reflection
        distance [chn,rd] = trigger
#        distance[chn,rd] = np.float16( (START_DELAY + trigger)*740.0/15000000)
        
print("Total Time is about ", time.time() - start_PROCESSING, "to read binary file and calliper")       
        
# del to gain more spaces
#del signal_matrices
        
class OneFileMainWindowData(object):
    
    def __init__(self, filename, mainWindowData):
        self.filename = filename
        self.mainWindowData = mainWindowData       
        
with open('save_result.pkl', 'wb') as output:
    save_result = OneFileMainWindowData(file_name, main_signal_matrices)
    pickle.dump(save_result, output, pickle.HIGHEST_PROTOCOL)

with open('save_result.pkl', 'rb') as input:
    read_pkl = pickle.load(input)

#def wt(data, keyname, wavefunc, level, m, n):
#    """
#    小波降噪函数
#    - level: 分解层数；
#    - data: 保存列表类型的字典；
#    - keyname: 键名；
#    - index_list: 待处理序列；
#    - wavefunc: 选取的小波函数；
#    - m,n 选择进行阈值处理的小波系数层数
#    """
#    # 分解
#    coeff = pywt.wavedec(data[keyname], wavefunc, mode='sym', level=level)
#    # 设置 sgn 函数
#    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0
#    # 降噪过程
#    for i in range(m, n + 1):  # 选取小波系数层数为 m~n 层
#        cD = coeff[i]
#        for j in range(len(cD)):
#            Tr = np.sqrt(2 * np.log(len(cD)))  # 计算阈值
#            if cD[j] >= Tr:
#                coeff[i][j] = sgn(cD[j]) - Tr  # 使用 sgn 函数向零收缩
#            else:
#                coeff[i][j] = 0  # 低于阈值置零
#    # 重新构建
#    denoised_data_list = my_wave_rec(coeff, wavefunc, data[keyname])
#    # 为了避免出现负值的情况，取绝对值
#    abs_denoised_list = list(map(lambda x: abs(x), denoised_data_list))
#    # 返回降噪结果
#    return abs_denoised_list

    