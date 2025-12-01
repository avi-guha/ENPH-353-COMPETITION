#!/usr/bin/env python3
import sys
import rospy
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy
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


# Create a panel with a LaTeX-style heading and fixed-size image display
def make_panel(title):
    container = QWidget()
    layout = QVBoxLayout()

    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_label.setStyleSheet("""
        font-family: 'CMU Serif';
        font-size: 18px;
        font-weight: bold;
    """)

    image_label = QLabel()
    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    image_label.setStyleSheet("background-color: black; color: white;")

    # FIXED image size (but panel can resize)
    image_label.setFixedSize(600, 400)

    layout.addWidget(title_label)
    layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)

    container.setLayout(layout)
    return container, image_label


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

        # Panels with headings
        self.raw_panel, self.raw_label = make_panel("Raw Board")
        self.proc_panel, self.proc_label = make_panel("Processed Board")
        self.words_panel, self.words_label = make_panel("Words Debug")
        self.letters_panel, self.letters_label = make_panel("Letters Debug")

        # Grid layout (2×2)
        grid = QGridLayout()
        grid.addWidget(self.raw_panel,    0, 0)
        grid.addWidget(self.proc_panel,   0, 1)
        grid.addWidget(self.words_panel,  1, 0)
        grid.addWidget(self.letters_panel,1, 1)

        widget = QWidget()
        widget.setLayout(grid)
        self.setCentralWidget(widget)

    # Update methods
    def update_raw(self, img):
        qimg = cv_to_qt(img)
        pix = QPixmap.fromImage(qimg)
        self.raw_label.setPixmap(
            pix.scaled(
                600, 500,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def update_proc(self, img):
        qimg = cv_to_qt(img)
        pix = QPixmap.fromImage(qimg)
        self.proc_label.setPixmap(
            pix.scaled(
                600, 500,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def update_words(self, img):
        qimg = cv_to_qt(img)
        pix = QPixmap.fromImage(qimg)
        self.words_label.setPixmap(
            pix.scaled(
                600, 500,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def update_letters(self, img):
        qimg = cv_to_qt(img)
        pix = QPixmap.fromImage(qimg)
        self.letters_label.setPixmap(
            pix.scaled(
                600, 500,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )


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
