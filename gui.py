#!/usr/bin/env python3
import sys
import rospy
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, Qt, QObject

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# Convert cv2 BGR to QImage
def cv_to_qt(img):
    if img is None:
        return QImage()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)


# Thread-safe signal bridge
class RosBridge(QObject):
    raw_sig = pyqtSignal(np.ndarray)
    proc_sig = pyqtSignal(np.ndarray)
    words_sig = pyqtSignal(np.ndarray)
    letters_sig = pyqtSignal(np.ndarray)


class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clueboard Debug Viewer")

        # Labels
        self.raw_label = QLabel("Raw Board")
        self.proc_label = QLabel("Processed Board")
        self.words_label = QLabel("Words Debug")
        self.letters_label = QLabel("Letters Debug")

        for lab in [self.raw_label, self.proc_label,
                    self.words_label, self.letters_label]:
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setStyleSheet("background-color: black; color: white;")

        # Layout
        top = QHBoxLayout()
        top.addWidget(self.raw_label)
        top.addWidget(self.proc_label)

        bottom = QVBoxLayout()
        bottom.addWidget(self.words_label)
        bottom.addWidget(self.letters_label)

        full = QVBoxLayout()
        full.addLayout(top)
        full.addLayout(bottom)

        widget = QWidget()
        widget.setLayout(full)
        self.setCentralWidget(widget)

    # Update methods
    def update_raw(self, img): 
        self.raw_label.setPixmap(QPixmap.fromImage(cv_to_qt(img)))

    def update_proc(self, img):
        self.proc_label.setPixmap(QPixmap.fromImage(cv_to_qt(img)))

    def update_words(self, img):
        self.words_label.setPixmap(QPixmap.fromImage(cv_to_qt(img)))

    def update_letters(self, img):
        self.letters_label.setPixmap(QPixmap.fromImage(cv_to_qt(img)))


class GuiNode:
    def __init__(self, bridge: RosBridge):
        self.bridge = bridge
        self.cv_bridge = CvBridge()

        rospy.init_node("clueboard_gui", anonymous=True)

        rospy.Subscriber("/clueboard/raw_board", Image,
                         self.cb_raw, queue_size=1)
        rospy.Subscriber("/clueboard/processed_board", Image,
                         self.cb_proc, queue_size=1)
        rospy.Subscriber("/clueboard/words_debug", Image,
                         self.cb_words, queue_size=1)
        rospy.Subscriber("/clueboard/letters_debug", Image,
                         self.cb_letters, queue_size=1)

    def cb_raw(self, msg):
        self.bridge.raw_sig.emit(self.cv_bridge.imgmsg_to_cv2(msg))

    def cb_proc(self, msg):
        self.bridge.proc_sig.emit(self.cv_bridge.imgmsg_to_cv2(msg))

    def cb_words(self, msg):
        self.bridge.words_sig.emit(self.cv_bridge.imgmsg_to_cv2(msg))

    def cb_letters(self, msg):
        self.bridge.letters_sig.emit(self.cv_bridge.imgmsg_to_cv2(msg))


def main():
    app = QApplication(sys.argv)

    bridge = RosBridge()
    gui = MainGUI()

    bridge.raw_sig.connect(gui.update_raw)
    bridge.proc_sig.connect(gui.update_proc)
    bridge.words_sig.connect(gui.update_words)
    bridge.letters_sig.connect(gui.update_letters)

    GuiNode(bridge)
    gui.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


