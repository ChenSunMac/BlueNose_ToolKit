# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:08:02 2017

@author: Chens
"""

import ftplib
import os
import socket
#HOST是远程FTP地址
HOST = '222.222.444.92'
DIRN = 'hcjy/css/'

def main():
    try:
        f = ftplib.FTP(HOST)
    except ftplib.error_perm:
        print('无法连接到"%s"' % HOST)
        return
    print('连接到"%s"' % HOST)
    
    try:
        #user是FTP用户名，pwd就是密码了
        f.login('user','pwd')
    except ftplib.error_perm:
        print('登录失败')
        f.quit()
        return
    print('登陆成功')
    
    try:
       #得到DIRN的工作目录
        f.cwd(DIRN)
    except ftplib.error_perm:
        print('列出当前目录失败')
        f.quit()
        return
    print(f.nlst())
  #f.nlst()返回一个当前目录下的列表返回给downloadlist
    downloadlist = f.nlst()
    try:
        os.getcwd()
       #创建一个css的同名文件夹
        os.mkdir('css')
      #切换到css文件夹，也就是改变当前工作目录，目的是为了将要下载的文件下载到这个文件夹
        os.chdir('css')
       #遍历刚才返回的文件名列表
        for FILE in downloadlist:
            f.retrbinary('RETR %s' % FILE,open(FILE,'wb').write)
            
            print('文件"%s"下载成功' % FILE)
    except ftplib.error_perm:
        print('无法读取"%s"' % FILE)
        os.unlink(FILE)
    else:
        print('文件全部下载完毕！')
        f.quit()
        return
    
    
    
if __name__ == '__main__':
    main() 