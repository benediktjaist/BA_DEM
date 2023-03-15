import sys
from PyQt5 import QtWidgets

def main():

    """
    allow you to get size of your current screen
    -1 is to precise that it is the current screen
    """
    sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
    print(" Screen size : "  + str(sizeObject.height()) + "x"  + str(sizeObject.width()))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main()
