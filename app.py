import sys
from torchvision import transforms
from PyQt5.QtWidgets import (QWidget, QPushButton, QMessageBox, QLabel, QApplication)
from PyQt5.QtGui import QIcon, QColor, QPixmap
from PyQt5 import QtWidgets
from PIL import Image
from utils import results_to_json, visualize_detections
from model import get_model


class Widget(QWidget):

	def __init__(self):
		super().__init__()
		self.initUI()
		self.model = None

	def initUI(self):

		self.setGeometry(300,300,500,300)
		col = QColor(255,255,255)
		self.setStyleSheet("QWidget { background-color: %s }"% col.name())
		self.setWindowTitle('GIS')
		self.setWindowIcon(QIcon('logo.jpg'))

		self.wb_path = "logo.jpg"
		self.pixmap = QPixmap("logo.jpg")
		self.lbl = QLabel(self)
		self.lbl.setPixmap(self.pixmap)
		self.lbl.move(88,16)

		pybutton = QPushButton('Найти объекты', self)
		load = QPushButton('Выбрать изображение', self)
		load_model = QPushButton('Выбрать модель', self)
		spravka = QPushButton('Справка', self)

		load.clicked.connect(self.click)
		pybutton.clicked.connect(self.clickMethod)
		load_model.clicked.connect(self.click_load_model)
		spravka.clicked.connect(self.clickspravka)

		pybutton.resize(150,30)
		load.resize(150,30)
		load_model.resize(150,30)
		spravka.resize(150, 30)

		pybutton.move(330,50)
		load.move(330,100)
		load_model.move(330,150)
		spravka.move(330, 200)

		self.show()

	def clickspravka(self):
		reference = "Программа для нахождения географических объектов изображению. Для начала нужно выбрать изображение и загрузить обученную модель, а затем нажать кнопку <<найти объекты>> для нахождения географических объектов. "
		QMessageBox.about(self,"Справка",reference)

	def clickMethod(self):
		if self.wb_path == "logo.jpg":
			QMessageBox.about(self, "Изображение не выбрано","Сначала нужно выбрать изображение")
			return
		if not self.model:
			QMessageBox.about(self, "Модель не выбрана", "Сначала нужно загрузить модель")
			return

		img = Image.open(str(self.wb_path)).convert('RGB')
		pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
		predictions = self.model(pil_to_tensor)[0]
		for key in predictions.keys():
			predictions[key] = predictions[key].detach().numpy()
		visualize_detections(img,
							 predictions['boxes'],
							 predictions['labels'],
							 predictions['scores'])
		result_path = "results/result.json"
		results_to_json(predictions, result_path)
		message = "Нахождение объектов завершено успешно. Файлы результатов сохранены в каталоге results"
		QMessageBox.about(self, "Нахождение объектов", message)

	def click_load_model(self):
		model_path = QtWidgets.QFileDialog.getOpenFileName()[0]
		if not model_path:
			QMessageBox.about(self, "Модель не выбрана", "Сначала нужно загрузить модель")
			return
		self.model = get_model(model_path)

	def click(self):
		wb = QtWidgets.QFileDialog.getOpenFileName()[0]
		self.wb_path = str(wb)

if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = Widget()
	sys.exit(app.exec_())