import os
import random
import time

import cv2
import numpy as np
from math import log10, copysign
from skimage import morphology
import image_slicer
from sklearn.neural_network import MLPClassifier


def area(element):
    return cv2.contourArea(element)


def simpleDetection(path, name):
    t = 4
    picture = cv2.imread(path + "/" + name, 0)
    kernel = np.ones((5, 5), np.uint8)
    a = picture
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    for i in range(2):
        a = clahe.apply(a)
        print("Clahe" + str(i))
        cv2.imshow('image', a)
        cv2.waitKey(t)

    a = opening = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    print("Opening")
    cv2.imshow('image', a)
    cv2.waitKey(t)

    for i in range(1):
        a = morphology.erosion(a)
        print("Erossion" + str(i))
        cv2.imshow('image', a)
        cv2.waitKey(t)

    thresh = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 8)
    print("threshold")
    cv2.imshow('image', thresh)
    cv2.waitKey(t)

    for i in range(2):
        thresh = morphology.erosion(thresh)
        print("Erossion" + str(i))
        cv2.imshow('image', thresh)
        cv2.waitKey(t)

    for i in range(1):
        thresh = morphology.dilation(thresh)
        print("Dilatation" + str(i))
        cv2.imshow('image', thresh)
        cv2.waitKey(t)

    """thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    print("Exit")
    cv2.imshow('image', thresh)
    cv2.waitKey(t"""

    thresh = cv2.bitwise_not(thresh)
    print("bitwise_not")
    cv2.imshow('image', thresh)
    cv2.waitKey(t)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursTab = []
    minArea = 100
    contours = sorted(contours, key=area)
    print("Contours count: ", len(contours))
    for contour in contours:
        # todo zmiana pola powierzchni na miare koloru
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        botmost = tuple(contour[contour[:, :, 1].argmax()][0])
        if cv2.contourArea(
                contour) > minArea and picture[leftmost[1]][leftmost[0]].any() > 0 and picture[rightmost[1]][
            rightmost[0]].any() > 0 and picture[topmost[1]][topmost[0]].any() > 0 and picture[botmost[1]][
            botmost[0]].any() > 0:
            contoursTab.append(contour)
        else:
            if cv2.contourArea(contour) > minArea:
                print("L ", picture[leftmost[1]][leftmost[0]])
                print("R ", picture[rightmost[1]][rightmost[0]])

    print("Contours count: ", len(contoursTab))
    cv2.drawContours(opening, contoursTab, -1, (255, 255, 255), 3)
    bitmap = np.zeros((len(opening), len(opening[0])))
    cv2.drawContours(bitmap, contoursTab, -1, (255, 255, 255), -1)
    cv2.imwrite("wyniki/" + path + "/" + name, opening)
    cv2.imwrite("mapy/" + path + "/" + name, bitmap)
    cv2.imshow('image', opening)
    cv2.waitKey(t)
    checkImg(path, name[:name.index(".")])


def checkImg(path, name, image=None, classification=False, ai=False):
    t = 4
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    orginalMap = cv2.imread(path + "_manualsegm/" + name + ".tif")
    if image is None:
        myMap = cv2.imread("mapy/" + path + "/" + name + ".jpg")
    else:
        myMap = image
    for i in range(len(orginalMap)):
        for j in range(len(orginalMap[0])):
            if i % 100 == 0 and j % 1000 == 0:
                print(i, " ", j)
            if myMap[i][j].all() != orginalMap[i][j].all():
                if orginalMap[i][j].all() != 0:
                    FN += 1
                    myMap[i][j] = [255, 0, 0]  # false negative

                else:
                    FP += 1
                    myMap[i][j] = [0, 0, 255]  # false positive
            else:
                if orginalMap[i][j].all() == 0:
                    TN += 1  # true negative
                else:
                    TP += 1  # true positive
    # todo zapis do pliku
    print("Count ", TP + TN + FP + FN)
    print("True positive: ", TP)
    print("True negative: ", TN)
    print("False negative: ", FN)
    print("False positive: ", FP)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("Acc: ", acc)
    sens = TP / (TP + FN)
    print("Sens: ", sens)
    spec = TN / (TN + FP)
    print("Spec: ", spec)
    cv2.imshow('image', myMap)
    if classification:
        path += "_classification"
    if ai:
        path += "_ai"
    try:
        os.makedirs("bledy/" + path)
    except FileExistsError:
        pass
    file = open("bledy/" + path + "/bledy.txt", "a+")
    wr = name + ";" + str(TP) + ";" + str(TN) + ";" + str(FN) + ";" + str(FP) + ";" + str(acc) + ";" + str(sens) + ";" + str(spec) + "\n"
    file.write(wr)
    file.close()
    cv2.imwrite("bledy/" + path + "/" + name + ".jpg", myMap)
    cv2.waitKey(t)


def loadData(path, name, n):
    tab = []
    for i in range(1, n + 1):
        full_name = "%02d" % i + name
        dane = open("splited/" + path + "/" + full_name + "/dane.txt")
        print("splited/" + path + "/" + full_name + "/dane.txt")
        i = 0
        for line in dane:
            if i == 0:
                i += 1
            else:
                a = line.replace("\n", "")
                sp = a.split(";")
                sp = sp[1:]
                sp = [float(i) for i in sp]
                # print(sp)
                tab.append(sp)
    return tab


def getFirst(tab):
    return tab[0]


def classify(slice, k, dane):
    r, g, b = average(slice)
    war = wariancja(r, g, b, slice)
    img_gray = cv2.cvtColor(slice.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    """moments1 = cv2.moments(im)
    mom = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11',
           'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
           'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
    moments = []
    for i in mom:
        moments.append(moments1[i])
    # huMoments = cv2.HuMoments(moments)"""
    """for i in range(len(moments)):
        try:
            huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
        except ValueError:
            huMoments[i] = 0"""
    d_tab = []
    for dana in dane:
        d = abs(r - dana[0]) + abs(g - dana[1]) + abs(b - dana[2]) + abs(war ** 0.5 - (dana[3] ** 0.5))
        """for i in range(len(moments)):
            d += abs(moments[i] - dana[4 + i])"""
        dec = dana[-1]
        d_tab.append([d, dec])
    d_tab = sorted(d_tab, key=getFirst)
    dec_b = 0
    dec_c = 0
    for i in range(k):
        if d_tab[i][1] == 0.0:
            dec_c += 1
        else:
            dec_b += 1
    if dec_b > dec_c:
        return 1
    else:
        return 0


def average(slice):
    r = 0
    g = 0
    b = 0
    for i in range(len(slice)):
        for j in range(len(slice[0])):
            r += slice[i][j][2]
            g += slice[i][j][1]
            b += slice[i][j][0]
    r /= (len(slice) * len(slice[0]))
    g /= (len(slice) * len(slice[0]))
    b /= (len(slice) * len(slice[0]))
    return r, g, b


def wariancja(r, g, b, slice):
    r_sum = 0
    g_sum = 0
    b_sum = 0
    for i in range(len(slice)):
        for j in range(len(slice[0])):
            r_sum += (r - slice[i][j][2]) ** 2
            g_sum += (g - slice[i][j][1]) ** 2
            b_sum += (b - slice[i][j][0]) ** 2
    return (r_sum + g_sum + b_sum) / (len(slice) * len(slice[0]) * 3)


def knn(path, name, n, k, size=5):
    start_time = time.time()
    img = cv2.imread(path + "/" + name)
    mapa = np.zeros((len(img), len(img[0]), 3))
    dane = loadData(path, "_h", n)
    for i in range(0, len(img), size):
        for j in range(0, len(img[0]), size):
            if i % 100 == 0 and j % 1000 == 0:
                print(i, " ", j)
                if j % 3000 == 0:
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
                    dec = classify(slice, k, dane)
                # print("Dec: ", dec)
                for a in range(size):
                    for b in range(size):
                        mapa[i + a][j + b] = [dec * 255, dec * 255, dec * 255]
    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    cv2.imshow('image', mapa)
    cv2.waitKey(0)
    checkImg(path, name[:name.index(".")], mapa, True)


def splitImg(path, name, n, size=5, tabB=[], tabC=[], begin=False):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.imread(path + "/" + name, 0)
    if begin:
        a = img
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        for _ in range(2):
            a = clahe.apply(a)
        img = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    name = name[:name.index(".")]
    mapa = cv2.imread(path + "_manualsegm/" + name + ".tif")
    count = 0
    biale = 0
    czarne = 0
    try:
        os.makedirs("splited/" + path + "/" + name)
    except FileExistsError:
        pass
    f = open("splited/" + path + "/" + name + "/dane.txt", "w+")
    f.write("Lp;R;G;B;War;m0;m1;m2;m3;m4;m5;m6;m7;Dec\n")
    print("Starting splitting")
    while count < n:
        # todo dodawnie czegoś co się nie powtarza
        i = random.randint(0, len(img))
        j = random.randint(0, len(img[0]))
        slice = np.zeros((size, size, 3))
        mapSlice = np.zeros((size, size, 3))
        if i + size < len(img) and j + size < len(img[i]):
            save = False
            for a in range(size):
                for b in range(size):
                    slice[a][b] = img[i + a][j + b]
                    mapSlice[a][b] = mapa[i + a][j + b]
                    if not save and slice[a][b][0] > 1 and slice[a][b][1] > 1 and slice[a][b][2] > 1:
                        # print(slice[a][b])
                        save = True
            if save:
                if mapSlice[size // 2][size // 2][0] > 0:
                    r, g, b = average(slice)
                    war = wariancja(r, g, b, slice)
                    img_gray = cv2.cvtColor(slice.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    _, im = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
                    moments = cv2.moments(im)
                    ##print(moments)
                    """huMoments = cv2.HuMoments(moments)"""
                    to_write = str(count) + ";" + str(r) + ";" + str(g) + ";" + str(b) + ";" + str(war)
                    mom = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11',
                           'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                           'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
                    for c in mom:
                        to_write += ";" + str(moments[c])

                    if save:
                        biale += 1
                        to_write += ";1\n"
                        f.write(to_write)
                        count += 1
                else:
                    save = random.randint(0, 5) == 0
                    if save:
                        r, g, b = average(slice)
                        war = wariancja(r, g, b, slice)
                        img_gray = cv2.cvtColor(slice.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                        _, im = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
                        moments = cv2.moments(im)
                        # print(moments)
                        """huMoments = cv2.HuMoments(moments)"""
                        to_write = str(count) + ";" + str(r) + ";" + str(g) + ";" + str(b) + ";" + str(war)
                        mom = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11',
                               'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                               'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
                        for c in mom:
                            to_write += ";" + str(moments[c])
                        if save:
                            czarne += 1
                            to_write += ";0\n"
                            f.write(to_write)
                            count += 1
                print("i = ", i, " j = ", j, " count = ", count)
    f.close()
    tabB.append(biale)
    tabC.append(czarne)


def aiClass(path, name, clf, size=5, begin=False):
    t = 2
    kernel = np.ones((5, 5), np.uint8)
    start_time = time.time()
    img = cv2.imread(path + "/" + name,0)
    if begin:
        a = img
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        for _ in range(2):
            a = clahe.apply(a)
        img = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    mapa = np.zeros((len(img), len(img[0]), 3))
    for i in range(0, len(img), size):
        for j in range(0, len(img[0]), size):
            if i % 100 == 0 and j % 1000 == 0:
                print(i, " ", j)
                if j % 3000 == 0:
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
                    X = []
                    r, g, b = average(slice)
                    X.append(r)
                    X.append(g)
                    X.append(b)
                    war = wariancja(r, g, b, slice)
                    X.append(war)
                    img_gray = cv2.cvtColor(slice.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    _, im = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
                    moments1 = cv2.moments(im)
                    mom = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11',
                           'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                           'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
                    for c in mom:
                        X.append(moments1[c])
                    if clf.predict([X])[0] > 0:
                        dec = 1
                    else:
                        dec = 0
                for a in range(size):
                    for b in range(size):
                        mapa[i + a][j + b] = [dec * 255, dec * 255, dec * 255]


    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    cv2.imshow('image', mapa)
    cv2.waitKey(t)
    checkImg(path, name[:name.index(".")], mapa, ai=True)


def lern(X, Y):
    clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
    for i in range(700):
        print("Learning", i)
        clf.fit(X, Y)
    return clf


def prepareData(data):
    X = []
    Y = []
    print(len(data[0]))
    for d in data:
        X.append(d[:-1])
        Y.append(int(d[-1]))
    return X, Y


if __name__ == '__main__':
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    begin  = True
    data1 = loadData("healthy", "_h", 10)# dla 9 dobre wyniki
    data2 = loadData("glaucoma", "_g", 10)
    data3 = []# loadData("diabetic_retinopathy", "_dr", 7)
    data = data1 + data2 + data3
    print("Len of data: ", len(data))
    X, Y = prepareData(data)
    clf = lern(X, Y)
    """file = open("bledy/glaucoma_ai/bledy.txt", "w+")
    wr = "name;TP;TN;FN;FP;acc;sens;spec\n"
    file.write(wr)
    file.close()"""
    names = os.listdir("glaucoma")
    for name in names[-1:]:
        aiClass("glaucoma", name, clf, begin=begin)

    names = os.listdir("healthy")
    for name in names[-1:]:
        aiClass("healthy", name, clf, begin=begin)


    #Simple detection
    """file = open("bledy/healthy/bledy.txt", "w+")
    wr = "name;TP;TN;FN;FP;acc;sens;spec\n"
    file.write(wr)
    file.close()
    names = os.listdir("healthy")
    for name in names:
        simpleDetection("healthy", name)"""

    """tabB = []
    tabC = []
    names = os.listdir("glaucoma")
    for name in names:
    #todo dodac wstępne przetwarzanue ibrazu
        splitImg("glaucoma", name, 500, tabB=tabB, tabC = tabC, begin=begin)
    # splitImg("healthy", name, 500, tabB=tabB, tabC=tabC)
    print("Białe: ", tabB, sum(tabB))
    print("Czarne: ", tabC, sum(tabC))"""

    #file = open("bledy/healthy_classification/bledy.txt", "w+")
    #wr = "name;TP;TN;FN;FP;acc;sens;spec"
    #file.write(wr)
    #knn("healthy", names[4], 2, 100)
    cv2.destroyAllWindows()
