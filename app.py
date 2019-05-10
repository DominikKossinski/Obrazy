import sys
import threading
import time

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog
from cv2 import cv2
from skimage import morphology

import obrazy


class App(QWidget):
    font = QFont('SansSerif', 10)
    label_width = 300
    text_area_width = 700
    img_size = 400
    size = 10
    begin = True

    def __init__(self, classifier):
        super().__init__()
        self.title = "DnoOka"
        self.clf = classifier
        self.width = 1600
        self.height = 900
        self.img_size = 450
        self.left = 10
        self.top = 10

        self.n = 1
        self.k = 101

        self.path = "healthy"
        self.name = "01_h.jpg"

        self.parent_widget = QWidget(self)
        self.parent_layout = QVBoxLayout()

        self.img_widget = QWidget()

        self.input_img = MyImage("Obraz wejściowy", "healthy/01_h.jpg", self.img_size)
        self.result_img = MyImage("Obraz wyjściowy", "mapy/healthy/01_h.jpg", self.img_size)
        self.error_img = MyImage("Błędy", "bledy/healthy/01_h.jpg", self.img_size)

        self.img_layout = QHBoxLayout()
        self.img_layout.addWidget(self.input_img)
        self.img_layout.addWidget(self.result_img)
        self.img_layout.addWidget(self.error_img)

        self.img_widget.setLayout(self.img_layout)
        self.parent_layout.addWidget(self.img_widget)

        self.second_widget = QWidget()
        self.second_layout = QHBoxLayout()

        self.buttons_widget = QWidget()
        self.buttons_layout = QVBoxLayout()

        # todo setup width
        self.select_button = QPushButton("Wybierz obraz")
        self.select_button.setFont(App.font)
        self.select_button.clicked.connect(self.change_img)
        self.buttons_layout.addWidget(self.select_button)

        self.simple_detection_button = QPushButton("Algorytm Podstawowy")
        self.simple_detection_button.setFont(App.font)
        self.simple_detection_button.clicked.connect(self.start_simple_detection)
        self.buttons_layout.addWidget(self.simple_detection_button)

        self.knn_class_button = QPushButton("Klasyfikator knn")
        self.knn_class_button.setFont(App.font)
        self.knn_class_button.clicked.connect(self.start_knn)
        self.buttons_layout.addWidget(self.knn_class_button)

        self.ai_class_button = QPushButton("Klasyfikator AI")
        self.ai_class_button.setFont(App.font)
        self.ai_class_button.clicked.connect(self.start_ai)
        self.buttons_layout.addWidget(self.ai_class_button)

        self.buttons_widget.setLayout(self.buttons_layout)

        self.second_layout.addWidget(self.buttons_widget)

        self.errors_widget = QWidget()
        self.errors_layout = QVBoxLayout()

        self.percent_label = QLabel("")
        self.percent_label.setFont(App.font)
        self.errors_layout.addWidget(self.percent_label)

        self.error_checking_label = QLabel("")
        self.error_checking_label.setFont(App.font)
        self.errors_layout.addWidget(self.error_checking_label)

        self.percent_labels = [self.percent_label, self.error_checking_label]

        self.acc_label = QLabel("Acc: ")
        self.acc_label.setFont(App.font)
        self.errors_layout.addWidget(self.acc_label)

        self.sens_label = QLabel("Sens: ")
        self.sens_label.setFont(App.font)
        self.errors_layout.addWidget(self.sens_label)

        self.spec_label = QLabel("Spec: ")
        self.spec_label.setFont(App.font)
        self.errors_layout.addWidget(self.spec_label)

        self.errors_labels = [self.acc_label, self.sens_label, self.spec_label]

        self.errors_widget.setLayout(self.errors_layout)

        self.second_layout.addWidget(self.errors_widget)

        self.second_widget.setLayout(self.second_layout)
        self.parent_layout.addWidget(self.second_widget)

        self.parent_widget.setLayout(self.parent_layout)

        self.init_ui()

    def change_img(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open File')[0]
        self.input_img.set_image(self.path)
        self.name = self.path[self.path.rindex("/") + 1:]
        self.path = self.path[:self.path.rindex("/")]
        self.path = self.path[self.path.rindex("/") + 1:]

    def start_simple_detection(self):
        self.reset_gui()
        thread = threading.Thread(
            target=simple_detection,
            args=(self.path, self.name, self.result_img, self.error_img, self.percent_labels, self.errors_labels,)
        )
        thread.start()

    def start_knn(self):
        self.reset_gui()
        thread = threading.Thread(
            target=knn,
            args=(self.path, self.name, self.n, self.k, self.result_img, self.error_img, self.percent_labels,
                  self.errors_labels,)
        )
        thread.start()

    def start_ai(self):
        self.reset_gui()
        self.result_img.set_img(np.zeros((self.img_size, self.img_size, 3)))
        self.error_img.set_img(np.zeros((self.img_size, self.img_size, 3)))
        thread = threading.Thread(
            target=start_ai,
            args=(
                self.path, self.name, self.result_img, self.error_img, self.percent_labels, self.errors_labels,
                self.clf, self.begin,)
        )
        thread.start()

    def reset_gui(self):
        self.result_img.set_img(np.zeros((self.img_size, self.img_size, 3)))
        self.error_img.set_img(np.zeros((self.img_size, self.img_size, 3)))
        for label in self.percent_labels:
            label.setText("")
        for label in self.errors_labels:
            label.setText("")

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()


class MyImage(QWidget):

    def __init__(self, text, path, img_size):
        super().__init__()
        self.layout = QVBoxLayout()
        self.img_size = img_size
        self.width = img_size + 35
        self.height = img_size + 35
        self.textLabel = QLabel()
        self.textLabel.setText(text)
        self.layout.addWidget(self.textLabel)
        self.textLabel.setFixedWidth(img_size + 35)
        self.textLabel.setFixedHeight(30)
        self.textLabel.setFont(QFont('SansSerif', 15))
        self.pixMapLabel = QLabel()
        self.layout.addWidget(self.pixMapLabel)
        self.pixMapLabel.setFixedWidth(img_size)
        self.pixMapLabel.setFixedHeight(img_size)
        self.set_image(path)

        self.setLayout(self.layout)

    def set_image(self, path, resize=True):
        pix_map = QPixmap(path)
        if resize:
            pix_map = pix_map.scaled(self.img_size, self.img_size, Qt.KeepAspectRatio)
        self.pixMapLabel.setPixmap(pix_map)

    def set_img(self, im, resize=True, name="none"):
        cv2.imwrite(name + ".png", im)
        pix_map = QPixmap(name + ".png")
        if resize:
            pix_map = pix_map.scaled(self.img_size, self.img_size, Qt.KeepAspectRatio)
        self.pixMapLabel.setPixmap(pix_map)


def simple_detection(path, name, result_image, error_image, percent_labels, errors_labels):
    picture = cv2.imread(path + "/" + name, 0)
    kernel = np.ones((5, 5), np.uint8)
    a = picture
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    for i in range(2):
        a = clahe.apply(a)
        print("Clahe" + str(i))
        result_image.set_img(a)

    a = opening = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    print("Opening")
    result_image.set_img(a)
    for i in range(1):
        a = morphology.erosion(a)
        print("Erosion" + str(i))
        result_image.set_img(a)

    thresh = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 8)
    print("threshold")
    result_image.set_img(thresh)

    for i in range(2):
        thresh = morphology.erosion(thresh)
        print("Erosion" + str(i))
        result_image.set_img(thresh)

    for i in range(1):
        thresh = morphology.dilation(thresh)
        print("Dilatation" + str(i))
        result_image.set_img(thresh)

    thresh = cv2.bitwise_not(thresh)
    print("bitwise_not")
    result_image.set_img(thresh)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_tab = []
    min_area = 100
    contours = sorted(contours, key=obrazy.area)
    print("Contours count: ", len(contours))
    for contour in contours:
        left_most = tuple(contour[contour[:, :, 0].argmin()][0])
        right_most = tuple(contour[contour[:, :, 0].argmax()][0])
        top_most = tuple(contour[contour[:, :, 1].argmin()][0])
        bottom_most = tuple(contour[contour[:, :, 1].argmax()][0])
        if cv2.contourArea(contour) > min_area and picture[left_most[1]][left_most[0]].any() > 0 and \
                picture[right_most[1]][right_most[0]].any() > 0 and picture[top_most[1]][top_most[0]].any() > 0 and \
                picture[bottom_most[1]][bottom_most[0]].any() > 0:
            contours_tab.append(contour)

    print("Contours count: ", len(contours_tab))
    bitmap = np.zeros((len(opening), len(opening[0]), 3))
    cv2.drawContours(bitmap, contours_tab, -1, (255, 255, 255), -1)
    result_image.set_img(bitmap)
    check_img(path, name[:name.index(".")], error_image, percent_labels[1], errors_labels, bitmap)


def knn(path, name, n, k, result_image, error_image, percent_labels, errors_labels, size=5, begin=True):
    start_time = time.time()
    kernel = np.ones((5, 5), np.uint8)
    if begin:
        img = cv2.imread(path + "/" + name, 0)
    else:
        img = cv2.imread(path + "/" + name)
    if begin:
        a = img
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        for _ in range(2):
            a = clahe.apply(a)
        img = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    bit_map = np.zeros((len(img), len(img[0]), 3))
    dane = obrazy.get_data(n, True)
    for i in range(0, len(img), size):
        for j in range(0, len(img[0]), size):
            if j % 1000 == 0:
                percent_labels[0].setText(
                    "Przetwarzenie: " + str(int((i * len(img[0]) + j) / (len(img) * len(img[0])) * 100)) + "%")
            if i % 100 == 0 and j % 1000 == 0:
                print(i, " ", j)
                if j % 3000 == 0:
                    result_image.set_img(bit_map)
                    elapsed_time = time.time() - start_time
                    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if i + size < len(img) and j + size < len(img[0]):
                slice = np.zeros((size, size, 3))
                c = 0
                for a in range(size):
                    for b in range(size):
                        slice[a][b] = img[i + a][j + b]
                        if slice[a][b][0] <= 5.0 and slice[a][b][1] <= 5.0 and slice[a][b][2] <= 5.0:
                            # print(slice[a][b])
                            c += 1
                if c == size ** 2:
                    dec = 0
                else:
                    dec = obrazy.classify(slice, k, dane)
                # print("Dec: ", dec)
                for a in range(size):
                    for b in range(size):
                        bit_map[i + a][j + b] = [dec * 255, dec * 255, dec * 255]
    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    result_image.set_img(bit_map)
    percent_labels[0].setText("Przetwarzenie: 100%")
    check_img(path, name[:name.index(".")], error_image, percent_labels[1], errors_labels, bit_map)


def check_img(path, name, error_image, percent_label, errors_labels, my_map):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    original_map = cv2.imread(path + "_manualsegm/" + name + ".tif")
    for i in range(len(original_map)):
        for j in range(len(original_map[0])):
            if j % 1000 == 0:
                percent_label.setText("Sprawdzenie: " + str(
                    int((i * len(original_map[0]) + j) / (len(original_map) * len(original_map[0])) * 100)) + "%")
            if i % 100 == 0 and j % 1000 == 0:
                print(i, " ", j)
            if my_map[i][j].all() != original_map[i][j].all():
                if original_map[i][j].all() != 0:
                    fn += 1
                    my_map[i][j] = [255, 0, 0]  # false negative

                else:
                    fp += 1
                    my_map[i][j] = [0, 0, 255]  # false positive
            else:
                if original_map[i][j].all() == 0:
                    tn += 1  # true negative
                else:
                    tp += 1  # true positive
    percent_label.setText("Sprawdzenie: 100%")
    print("Count ", tp + tn + fp + fn)
    print("True positive: ", tp)
    print("True negative: ", tn)
    print("False negative: ", fn)
    print("False positive: ", fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    errors_labels[0].setText("Acc: " + str(acc))
    print("Acc: ", acc)
    sens = tp / (tp + fn)
    errors_labels[1].setText("Sens: " + str(sens))
    print("Sens: ", sens)
    spec = tn / (tn + fp)
    print("Spec: ", spec)
    errors_labels[2].setText("Spec: " + str(spec))
    error_image.set_img(my_map)


def start_ai(path, name, result_image, error_image, percent_labels, error_labels, classifier, begin):
    ai_class(path, name, classifier, result_image, error_image, percent_labels, error_labels, begin=begin)


def ai_class(path, name, classifier, result_image, error_image, percent_labels, error_labels, size=10, begin=False):
    kernel = np.ones((5, 5), np.uint8)
    start_time = time.time()
    img = cv2.imread(path + "/" + name, 0)
    if begin:
        a = img
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        for _ in range(2):
            a = clahe.apply(a)
        img = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    bit_map = np.zeros((len(img), len(img[0]), 3))
    for i in range(0, len(img), size):
        for j in range(0, len(img[0]), size):
            if j % 1000 == 0:
                percent_labels[0].setText(
                    "Przetwarzenie: " + str(int((i * len(img[0]) + j) / (len(img) * len(img[0])) * 100)) + "%")
            if i % 100 == 0 and j % 1000 == 0:
                print(i, " ", j)
                if j % 3000 == 0 and j != 0:
                    result_image.set_img(bit_map)
                    elapsed_time = time.time() - start_time
                    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if i + size < len(img) and j + size < len(img[0]):
                slice = np.zeros((size, size, 3))
                c = 0
                for a in range(size):
                    for b in range(size):
                        slice[a][b] = img[i + a][j + b]
                        if slice[a][b][0] <= 5.0 and slice[a][b][1] <= 5.0 and slice[a][b][2] <= 5.0:
                            # print(slice[a][b])
                            c += 1
                if c == size ** 2:
                    dec = 0
                else:
                    x = []
                    r, g, b = obrazy.average(slice)
                    x.append(r)
                    x.append(g)
                    # X.append(b)
                    war = obrazy.variance(r, g, b, slice)
                    x.append(war)
                    img_gray = cv2.cvtColor(slice.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    _, im = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
                    moments1 = cv2.moments(im)
                    hu_moments = cv2.HuMoments(moments1)
                    """mom = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11',
                           'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                           'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
                    for c in mom:
                        x.append(moments1[c])"""
                    for c in hu_moments:
                        x.append(c[0])
                    if classifier.predict([x])[0] > 0:
                        dec = 1
                    else:
                        dec = 0
                for a in range(size):
                    for b in range(size):
                        bit_map[i + a][j + b] = [dec * 255, dec * 255, dec * 255]

    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    result_image.set_img(bit_map)
    percent_labels[0].setText("Przetwarzenie: 100%")
    check_img(path, name[:name.index(".")], error_image, percent_labels[1], error_labels, bit_map)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '0':
            obrazy.generate_data(2000, App.size, App.begin, False)
    clf = obrazy.get_clf()
    app = QApplication(sys.argv)
    ex = App(clf)
    sys.exit(app.exec_())
