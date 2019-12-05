import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import time
from Problem_2.image_utilities import *

class Canvas(QLabel):
	def __init__(self):
		super().__init__()
		# Placeholder
		pixmap = QPixmap(500, 300)
		self.setPixmap(pixmap)
		self.draw_Lines = True
		self.pen_color = (0,0,0)
		self.last_x = None
		self.last_y = None
		self.filename = ""

	def mouseMoveEvent(self, e):
		if (self.draw_Lines == False):
			return  # Do nothing
		if self.last_x is None:  # First event.
			self.last_x = e.x()
			self.last_y = e.y()
			return  # Ignore the first time.

		painter = QPainter(self.pixmap())
		pen = painter.pen()
		pen.setWidth(8)
		r,g,b = self.pen_color
		pen.setColor(QColor(r,g,b))
		painter.setPen(pen)
		painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
		painter.end()
		self.update()
		self.last_x = e.x()
		self.last_y = e.y()

	def mousePressEvent(self, e):
		if (self.draw_Lines == True):
			return
		if self.last_x is None:  # First event.
			self.last_x = e.x()
			self.last_y = e.y()
			return  # Ignore the first time.

		painter = QPainter(self.pixmap())
		pen = painter.pen()
		pen.setWidth(8)
		r, g, b = self.pen_color
		pen.setColor(QColor(r, g, b))
		painter.setPen(pen)
		painter.drawRect(QRect(QPoint(self.last_x, self.last_y), QPoint(e.x(), e.y())))
		painter.end()
		self.update()
		self.last_x = None
		self.last_y = None

	def mouseReleaseEvent(self, e):
		if not self.draw_Lines:
			return
		self.last_x = None
		self.last_y = None

	def canvas_setup(self, filename, height, width):
		self.filename = filename
		pixmap = QPixmap(filename).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		self.setPixmap(pixmap)

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		# menu
		menu = self.menuBar()
		file_menu = menu.addMenu("&File")

		open_file_action = QAction("Open", self)
		open_file_action.setStatusTip("Open an image...")
		open_file_action.triggered.connect(self.openimage)
		file_menu.addAction(open_file_action)
		file_menu.addSeparator()

		save_file_action = QAction("Save", self)
		save_file_action.setStatusTip("Save an image...")
		save_file_action.triggered.connect(self.saveimage)
		file_menu.addAction(save_file_action)
		file_menu.addSeparator()

		# toolbar
		toolbar = QToolBar("Main toolbar")
		toolbar.setIconSize(QSize(16, 16))
		self.addToolBar(toolbar)
		foreground_action = QAction("Foreground", self)
		foreground_action.setStatusTip("Click to select foreground brush (green)")
		foreground_action.triggered.connect(self.foregroundBrushEvent)
		toolbar.addAction(foreground_action)

		toolbar.addSeparator()

		background_action = QAction("Background", self)
		background_action.setStatusTip("Click to select background brush (red)")
		background_action.triggered.connect(self.backgroundBrushEvent)
		toolbar.addAction(background_action)

		toolbar.addSeparator()

		garbage_mask_action = QAction("Garbage Mask", self)
		garbage_mask_action.setStatusTip("Create garbage mask")
		garbage_mask_action.triggered.connect(self.garbageMaskEvent)
		toolbar.addAction(garbage_mask_action)

		toolbar.addSeparator()

		segment_action = QAction("Segment", self)
		segment_action.setStatusTip("Create the segment")
		segment_action.triggered.connect(self.segmentEvent)
		toolbar.addAction(segment_action)

		self.setStatusBar(QStatusBar(self))

		# drawing/canvas
		self.canvas = Canvas()
		w = QWidget()
		l = QVBoxLayout()
		w.setLayout(l)
		l.addWidget(self.canvas)
		self.setCentralWidget(w)

	def foregroundBrushEvent(self, e):
		self.canvas.pen_color = (0,255,0)
		self.canvas.draw_Lines = True

	def backgroundBrushEvent(self, e):
		self.canvas.pen_color = (255,0,0)
		self.canvas.draw_Lines = True

	def garbageMaskEvent(self, e):
		self.canvas.pen_color = (0,0,255)
		self.canvas.draw_Lines = False

	def segmentEvent(self, e):
		if self.canvas.filename != "":
			image = self.canvas.pixmap().toImage()
			segment_image(image, self.canvas.filename)

	def openimage(self, s):
		print("click", s)

		filename, _ = QFileDialog.getOpenFileName(self, "Open file", "",
												  "PNG, JPG (*.png *.jpg);;"
												  "All files (*.*)")
		# filename = "./tmp/input_image.png"
		if filename:
			image = cv2.imread(filename)
			image = image_resize(image)
			height, width, _ = image.shape
			self.canvas.canvas_setup(filename, height, width)

	def saveimage(self, s):
		print("click", s)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
