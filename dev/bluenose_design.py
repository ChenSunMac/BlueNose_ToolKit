# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bluenose_design.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.DistanceMap = QtWidgets.QGraphicsView(self.centralwidget)
        self.DistanceMap.setGeometry(QtCore.QRect(410, 60, 256, 192))
        self.DistanceMap.setObjectName("DistanceMap")
        self.FileNames = QtWidgets.QTextBrowser(self.centralwidget)
        self.FileNames.setGeometry(QtCore.QRect(30, 60, 256, 192))
        self.FileNames.setObjectName("FileNames")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_Bin_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_Bin_File.setObjectName("actionOpen_Bin_File")
        self.actionOpen_Folder = QtWidgets.QAction(MainWindow)
        self.actionOpen_Folder.setObjectName("actionOpen_Folder")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionExport_Report = QtWidgets.QAction(MainWindow)
        self.actionExport_Report.setObjectName("actionExport_Report")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionAnalyzer_Help = QtWidgets.QAction(MainWindow)
        self.actionAnalyzer_Help.setObjectName("actionAnalyzer_Help")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.menuFile.addAction(self.actionOpen_Bin_File)
        self.menuFile.addAction(self.actionOpen_Folder)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExport_Report)
        self.menuFile.addAction(self.actionQuit)
        self.menuHelp.addAction(self.actionAnalyzer_Help)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Blue Nose Analyzer"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen_Bin_File.setText(_translate("MainWindow", "Open Bin Files"))
        self.actionOpen_Folder.setText(_translate("MainWindow", "Open Folder"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionExport_Report.setText(_translate("MainWindow", "Export Report"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionAnalyzer_Help.setText(_translate("MainWindow", "Analyzer Help"))
        self.actionAbout.setText(_translate("MainWindow", "About"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

