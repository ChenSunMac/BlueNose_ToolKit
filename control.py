# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:21:47 2017

@author: Chens
"""

import sys
from PyQt5.QtWidgets import (QWidget, QToolTip, 
    QPushButton, QApplication, QDesktopWidget)

from PyQt5.QtGui import QFont    


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):
        # Set font for the UI 
        QToolTip.setFont(QFont('SansSerif', 10))
        
        self.setToolTip('This is a <b>QWidget</b> widget')

        btn = QPushButton('Button', self)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        btn.resize(btn.sizeHint())
        btn.move(50, 50)       
        
        self.setGeometry(300, 300, 400, 400)
        self.setWindowTitle('Tooltips')    
        self.center()
        self.show()

    """
    Make the GUI appear at the Center of the Screen
    - center(self)
    """
    def center(self):
        # Find the geometry of the main screen 
        qr = self.frameGeometry() 
        # Find the resolution of the screen and the left top point position
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())



if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())