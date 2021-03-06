# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:59:45 2017

@author: chens
"""

import sys
import Algorithms.AlgorithmSet as AlgSet

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QToolTip, 
                             QPushButton,QDesktopWidget, QAction,
                             QProgressBar, QCheckBox, QMessageBox,
                             QTextEdit, QFileDialog, QSizePolicy)

import random


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 500, 700)
        self.setWindowTitle('BlueNose Analyzer')
        self.setWindowIcon(QtGui.QIcon('Kakyoin.png'))

        self.data = DataHolder()
        
        extractAction = QAction("&GET TO THE CHOPPAH!!!", self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip('Leave The App')
        extractAction.triggered.connect(self.close_application)

        openEditor = QAction("&Editor", self)
        openEditor.setShortcut("Ctrl+E")
        openEditor.setStatusTip('Open Editor')
        openEditor.triggered.connect(self.editor)

        openFile = QAction("&Open File", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(self.file_open)

        self.statusBar()

        mainMenu = self.menuBar()
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
        fileMenu.addAction(openFile)
        
        editorMenu = mainMenu.addMenu("&Editor")
        editorMenu.addAction(openEditor)


        

        
        
        self.center()
        self.home()
                
    def home(self):
        btn = QPushButton("Quit", self)
        btn.clicked.connect(self.close_application)
        btn.resize(btn.minimumSizeHint())
        btn.move(0,100)
        ### TooL Bar
        extractAction = QAction(QtGui.QIcon('todachoppa.png'), 'Flee the Scene', self)
        extractAction.triggered.connect(self.close_application)
        self.toolBar = self.addToolBar("Extraction")
        self.toolBar.addAction(extractAction)
        
        openBinFile = QAction(QtGui.QIcon('bin_icon.png'), 'Open Bin File', self)
        openBinFile.triggered.connect(self.process_bin_file)
        self.toolBar = self.addToolBar("Open Bin File")
        self.toolBar.addAction(openBinFile)        
        
        ### CHECK BOX
        checkBox = QCheckBox('Shrink Window', self)
        checkBox.move(100, 25)
        checkBox.stateChanged.connect(self.enlarge_window)
        
        ### Progress Bar and BUTTON
        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        
        self.btn = QPushButton("Download",self)
        self.btn.move(200,120)
        self.btn.clicked.connect(self.download)
        ### Plot Canvas
        
        self.canvas = PlotCanvas(self, width=5, height=4)
        self.canvas.move(200,200)

        self.show()


    def download(self):
        self.completed = 0

        while self.completed < 100:
            self.completed += 0.0001
            self.progress.setValue(self.completed)
        
        

    def enlarge_window(self, state):
        if state == QtCore.Qt.Checked:
            self.setGeometry(50,50, 1000, 800)
        else:
            self.setGeometry(50, 50, 500, 700)
        self.center()
        
    def center(self):
        # Find the geometry of the main screen 
        qr = self.frameGeometry() 
        # Find the resolution of the screen and the left top point position
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def editor(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)

    def file_open(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        file = open(name[0],'r')

        self.editor()

        with file:
            text = file.read()
            self.textEdit.setText(text)


    def process_bin_file(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open File')
        with open(file_name[0], "rb") as bin_file:
            signal_matrices = AlgSet.processBinFile(bin_file)
        
        #self.editor()
        #text = str(signal_matrices[0,0,:])
        #self.textEdit.setText(text)
        distanceMap = AlgSet.calliperAlg(signal_matrices)
        #self.canvas.plot_data(signal_matrices[0,0,:])
        self.canvas.plot_data(distanceMap)
        
    def close_application(self):
        choice = QMessageBox.question(self, 'Extract!',
                                            "Get into the chopper?",
                                            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Extracting Naaaaaaoooww!!!!")
            sys.exit()
        else:
            pass
  

class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        #self.plot()
 
 
    def plot(self):
        data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        ax.set_title('PyQt Matplotlib Example')
        self.draw()      

    def plot_data(self,data):
        #ax = self.figure.add_subplot(111)
        fig = plt.figure()
        plt.imshow(data, aspect='auto');
        plt.colorbar()
        plt.show()
#        ax.plot(data, 'r-')
#        ax.set_title('One single 2000 point signal')
        fig.show()
        self.draw()         


class DataHolder(object):
    """
    Hold data for plot and other interfaces.
    
    """
    def __init__(self):
        self.binFileNames = []
        self.dirName = []
        self.dataToPlot = None
        
        
def run():
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())


run()