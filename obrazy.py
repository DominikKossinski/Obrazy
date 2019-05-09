import os
import random
import time

import cv2
import numpy as np
from skimage import morphology
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def area(element):
    return cv2.contourArea(element)


def simple_detection(path, file_name):
    t = 4
    picture = cv2.imread(path + "/" + file_name, 0)
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
        print("Erosion" + str(i))
        cv2.imshow('image', a)
        cv2.waitKey(t)

    thresh = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 8)
    print("threshold")
    cv2.imshow('image', thresh)
    cv2.waitKey(t)

    for i in range(2):
        thresh = morphology.erosion(thresh)
        print("Erosion" + str(i))
        cv2.imshow('image', thresh)
        cv2.waitKey(t)

    for i in range(1):
        thresh = morphology.dilation(thresh)
        print("Dilatation" + str(i))
        cv2.imshow('image', thresh)
        cv2.waitKey(t)

    thresh = cv2.bitwise_not(thresh)
    print("bitwise_not")
    cv2.imshow('image', thresh)
    cv2.waitKey(t)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_tab = []
    min_area = 100
    contours = sorted(contours, key=area)
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
    cv2.drawContours(picture, contours_tab, -1, (255, 255, 255), 3)
    bitmap = np.zeros((len(opening), len(opening[0]), 3))
    cv2.drawContours(bitmap, contours_tab, -1, (255, 255, 255), -1)
    try:
        os.makedirs("maps/" + path)
    except FileExistsError:
        pass
    cv2.imwrite("maps/" + path + "/" + file_name, bitmap)
    cv2.imshow('image', picture)
    cv2.waitKey(t)
    check_img(path, file_name[:file_name.index(".")], bitmap)


def check_img(path, file_name, my_map, classification=False, ai=False):
    t = 4
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    original_map = cv2.imread(path + "_manualsegm/" + file_name + ".tif")
    for i in range(len(original_map)):
        for j in range(len(original_map[0])):
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
    print("Count ", tp + tn + fp + fn)
    print("True positive: ", tp)
    print("True negative: ", tn)
    print("False negative: ", fn)
    print("False positive: ", fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    print("Acc: ", acc)
    sens = tp / (tp + fn)
    print("Sens: ", sens)
    spec = tn / (tn + fp)
    print("Spec: ", spec)
    cv2.imshow('image', my_map)
    if classification:
        path += "_classification"
    if ai:
        path += "_ai"
    try:
        os.makedirs("errors/" + path)
    except FileExistsError:
        pass
    errors_file = open("errors/" + path + "/errors.txt", "a+")
    to_write = file_name + ";" + str(tp) + ";" + str(tn) + ";" + str(fn) + ";" + str(fp) + ";" + str(acc) + ";" + str(
        sens) + ";" + str(spec) + "\n"
    errors_file.write(to_write)
    errors_file.close()
    cv2.imwrite("errors/" + path + "/" + file_name + ".jpg", my_map)
    cv2.waitKey(t)


def load_data(path, file_name, n, is_knn):
    tab = []
    if is_knn:
        path += "_knn"
    for i in range(1, n + 1):
        full_name = "%02d" % i + file_name
        dane = open("split/" + path + "/" + full_name + "/dane.txt")
        print("split/" + path + "/" + full_name + "/dane.txt")
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


def get_first(tab):
    return tab[0]


def classify(img_slice, k, dane):
    r, g, b = average(img_slice)
    war = variance(r, g, b, img_slice)
    """img_gray = cv2.cvtColor(img_slice.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    moments1 = cv2.moments(im)
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
    d_tab = sorted(d_tab, key=get_first)
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


def average(img_slice):
    r = 0
    g = 0
    b = 0
    for i in range(len(img_slice)):
        for j in range(len(img_slice[0])):
            r += img_slice[i][j][2]
            g += img_slice[i][j][1]
            b += img_slice[i][j][0]
    r /= (len(img_slice) * len(img_slice[0]))
    g /= (len(img_slice) * len(img_slice[0]))
    b /= (len(img_slice) * len(img_slice[0]))
    return r, g, b


def variance(r, g, b, img_slice):
    r_sum = 0
    g_sum = 0
    b_sum = 0
    for i in range(len(img_slice)):
        for j in range(len(img_slice[0])):
            r_sum += (r - img_slice[i][j][2]) ** 2
            g_sum += (g - img_slice[i][j][1]) ** 2
            b_sum += (b - img_slice[i][j][0]) ** 2
    return (r_sum + g_sum + b_sum) / (len(img_slice) * len(img_slice[0]) * 3)


def knn(path, file_name, n, k, begin_transform, size=5):
    t = 4
    kernel = np.ones((5, 5), np.uint8)
    start_time = time.time()
    if begin_transform:
        img = cv2.imread(path + "/" + file_name, 0)
    else:
        img = cv2.imread(path + "/" + file_name)
    if begin_transform:
        a = img
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        for _ in range(2):
            a = clahe.apply(a)

        img = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    bit_map = np.zeros((len(img), len(img[0]), 3))
    dane = get_data(n, True)
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
                        bit_map[i + a][j + b] = [dec * 255, dec * 255, dec * 255]
    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    cv2.imshow('image', bit_map)
    try:
        os.makedirs("maps/" + path + "_classification")
    except FileExistsError:
        pass
    cv2.imwrite("maps/" + path + "_classification/" + file_name, bit_map)
    cv2.waitKey(t)
    check_img(path, file_name[:file_name.index(".")], bit_map, classification=True)


def split_img(path, file_name, n, size=5, tab_w=None, tab_b=None, begin_transform=False, is_knn=True):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.imread(path + "/" + file_name, 0)
    if begin_transform:
        a = img
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        for _ in range(2):
            a = clahe.apply(a)

        img = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    file_name = file_name[:file_name.index(".")]
    bit_map = cv2.imread(path + "_manualsegm/" + file_name + ".tif")
    count = 0
    white = 0
    black = 0
    if is_knn:
        path += "_knn"
    try:
        os.makedirs("split/" + path + "/" + file_name)
    except FileExistsError:
        pass
    f = open("split/" + path + "/" + file_name + "/dane.txt", "w+")
    if is_knn:
        f.write("Lp;R;G;B;War;Dec\n")
    else:
        f.write("Lp;R;G;War;m0;m1;m2;m3;m4;m5;Dec\n")
    print("Starting splitting")
    while count < n:
        i = random.randint(0, len(img))
        j = random.randint(0, len(img[0]))
        slice = np.zeros((size, size, 3))
        map_slice = np.zeros((size, size, 3))
        if i + size < len(img) and j + size < len(img[i]):
            save = False
            for a in range(size):
                for b in range(size):
                    slice[a][b] = img[i + a][j + b]
                    map_slice[a][b] = bit_map[i + a][j + b]
                    if not save and slice[a][b][0] > 1 and slice[a][b][1] > 1 and slice[a][b][2] > 1:
                        save = True
            if save:
                if map_slice[size // 2][size // 2][0] > 0:
                    r, g, b = average(slice)
                    war = variance(r, g, b, slice)
                    if is_knn:
                        to_write = str(count) + ";" + str(r) + ";" + str(g) + ";" + str(b) + ";" + str(war)
                    else:
                        img_gray = cv2.cvtColor(slice.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                        _, im = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
                        moments = cv2.moments(im)
                        hu_moments = cv2.HuMoments(moments)
                        to_write = str(count) + ";" + str(r) + ";" + str(g) + ";" + str(war)
                        # to_write = str(count) + ";" + str(r) + ";" + str(g) + ";" + str(b) + ";" + str(war)
                        """mom = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11',
                               'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                               'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
                        for c in mom:
                            to_write += ";" + str(moments[c])"""
                        for c in hu_moments:
                            to_write += ";" + str(c[0])
                    if save:
                        white += 1
                        to_write += ";1\n"
                        f.write(to_write)
                        count += 1
                else:
                    save = random.randint(0, 5) == 0
                    if save:
                        r, g, b = average(slice)
                        war = variance(r, g, b, slice)
                        if is_knn:
                            to_write = str(count) + ";" + str(r) + ";" + str(g) + ";" + str(b) + ";" + str(war)
                        else:
                            img_gray = cv2.cvtColor(slice.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                            _, im = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
                            moments = cv2.moments(im)
                            hu_moments = cv2.HuMoments(moments)
                            to_write = str(count) + ";" + str(r) + ";" + str(g) + ";" + str(war)
                            # to_write = str(count) + ";" + str(r) + ";" + str(g) + ";" + str(b) + ";" + str(war)
                            """mom = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11',
                                   'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                                   'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
                            for c in mom:
                                to_write += ";" + str(moments[c])"""
                            for c in hu_moments:
                                to_write += ";" + str(c[0])
                        if save:
                            black += 1
                            to_write += ";0\n"
                            f.write(to_write)
                            count += 1
                print("i = ", i, " j = ", j, " count = ", count)
    f.close()
    if tab_w is not None:
        tab_w.append(white)
    if tab_b is not None:
        tab_b.append(black)


def ai_class(path, file_name, classifier, size=5, begin_transform=False):
    t = 2
    kernel = np.ones((5, 5), np.uint8)
    start_time = time.time()
    img = cv2.imread(path + "/" + file_name, 0)
    if begin_transform:
        a = img
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        for _ in range(2):
            a = clahe.apply(a)
        img = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    bit_map = np.zeros((len(img), len(img[0]), 3))
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
                            c += 1
                if c == size ** 2:
                    dec = 0
                else:
                    x = []
                    r, g, b = average(slice)
                    x.append(r)
                    x.append(g)
                    # x.append(b)
                    war = variance(r, g, b, slice)
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
    cv2.imshow('image', bit_map)
    cv2.waitKey(t)
    try:
        os.makedirs("maps/" + path + "_ai")
    except FileExistsError:
        pass
    cv2.imwrite("maps/" + path + "_ai/" + file_name, bit_map)
    check_img(path, file_name[:file_name.index(".")], bit_map, ai=True)


def learn(x, y, steps=15):
    classifier = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', random_state=1, max_iter=1000,
                               warm_start=True)
    learning = True
    i = 1
    while learning:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        for j in range(steps):
            classifier.fit(x_train, y_train)
        print("Learning", i, " ", i * steps)
        i += 1
        score = classifier.score(x_test, y_test)
        print("Score:", score)
        learning = score < 0.843
    return classifier


def prepare_data(all_data):
    x = []
    y = []
    for d in all_data:
        x.append(d[:-1])
        y.append(int(d[-1]))
    return x, y


def get_data(n, is_knn):
    data_1 = load_data("healthy", "_h", n, is_knn)
    data_2 = load_data("glaucoma", "_g", n, is_knn)
    data_3 = load_data("diabetic_retinopathy", "_dr", n, is_knn)
    all_data = data_1 + data_2 + data_3
    print("Data set length: ", len(all_data))
    return all_data


def get_clf():
    all_data = get_data(10, False)
    x_data, y_data = prepare_data(all_data)
    return learn(x_data, y_data)


def generate_data(n, size, begin_transform, is_knn):
    tab_b = []
    tab_c = []
    file_names = os.listdir("healthy")
    for file_name in file_names:
        split_img("healthy", file_name, n, size=size, tab_w=tab_b, tab_b=tab_c, begin_transform=begin_transform, is_knn=is_knn)
    file_names = os.listdir("glaucoma")
    for file_name in file_names:
        split_img("glaucoma", file_name, n, size=size, tab_w=tab_b, tab_b=tab_c, begin_transform=begin_transform, is_knn=is_knn)
    file_names = os.listdir("diabetic_retinopathy")
    for file_name in file_names:
        split_img("diabetic_retinopathy", file_name, n, size=size, tab_w=tab_b, tab_b=tab_c,
                  begin_transform=begin_transform, is_knn=is_knn)
    print("White: ", tab_b, sum(tab_b))
    print("Black: ", tab_c, sum(tab_c))


if __name__ == '__main__':
    directories = ["healthy", "glaucoma", "diabetic_retinopathy"]
    clf = get_clf()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    """print("\nsimple\n")
    for directory in directories[]:
        files = os.listdir(directory)
        try:
            os.makedirs("errors/" + directory)
        except FileExistsError:
            pass
        #errors = open("errors/" + directory + "/errors.txt", "w+")
        #wr = "name;TP;TN;FN;FP;acc;sens;spec\n"
        #errors.write(wr)
        #errors.close()
        for file in files:
            print(file)
            simple_detection(directory, file)"""

    #todo przeliczyć dla [10:11] dla wszystkich folderów
    print("\nai\n")
    for directory in directories:
        files = os.listdir(directory)
        try:
            os.makedirs("errors/" + directory + "_ai")
        except FileExistsError:
            pass
        #errors = open("errors/" + directory + "_ai/errors.txt", "w+")
        #wr = "name;TP;TN;FN;FP;acc;sens;spec\n"
        #errors.write(wr)
        #errors.close()
        for file in files[10:11]:
            print(file)
            ai_class(directory, file, clf, 10, True)

    #todo przeliczyć
    """print("\nKnn\n")
    for directory in directories:
        files = os.listdir(directory)
        try:
            os.makedirs("errors/" + directory + "_classification")
        except FileExistsError:
            pass
        #errors = open("errors/" + directory + "_classification/errors.txt", "w+")
        #wr = "name;TP;TN;FN;FP;acc;sens;spec\n"
        #errors.write(wr)
        #errors.close()
        for file in files[-5:]:
            print(file)
            knn(directory, file, 1, 100, 10)"""

    cv2.destroyAllWindows()
