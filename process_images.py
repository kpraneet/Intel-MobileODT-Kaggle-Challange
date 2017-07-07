import os
import cv2
import math
import numpy as np
import time
import pickle


from PIL import ImageFile
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure


ImageFile.LOAD_TRUNCATED_IMAGES = True


def findArea(val):
    area = (0, 0, 0)
    height = []
    position = []
    for x in range(len(val)):
        if len(height) == 0:
            if val[x] > 0:
                height.append(val[x])
                position.append(x)
        else:
            if val[x] > height[-1]:
                height.append(val[x])
                position.append(x)
            elif val[x] < height[-1]:
                lPosition = None
                while val[x] < height[-1]:
                    maxH = height.pop()
                    calArea = maxH * (x - position[-1])
                    if calArea > area[0]:
                        area = (calArea, position[-1], x)
                    lPosition = position.pop()
                    if len(height) == 0:
                        break
                position.append(lPosition)
                if len(height) == 0:
                    height.append(val[x])
                elif height[-1] < val[x]:
                    height.append(val[x])
                else:
                    position.pop()
    while len(height) > 0:
        maxH = height.pop()
        lPosition = position.pop()
        calArea = maxH * (len(val) - lPosition)
        if calArea > area[0]:
            area = (calArea, len(val), lPosition)
    return area


def findRectangle(img):
    area = (0, 0, 0)
    addVal = np.zeros(img.shape)
    for x in range(img.shape[0]):
        if x == 0:
            addVal[x] = img[x]
            calArea = findArea(addVal[x])
            if calArea[0] > area[0]:
                area = calArea + (x, )
        else:
            addVal[x] = img[x] + addVal[x-1]
            addVal[x][img[x] == 0] *= 0
            calArea = findArea(addVal[x])
            if calArea[0] > area[0]:
                area = calArea + (x, )
    returnVal = int(area[3] + 1 - area[0] / abs(area[1] - area[2]))
    return returnVal, area[2], area[3], area[1], area[0]


def findCircle(img):
    if img.shape[0] > img.shape[1]:
        cropSize = (int(img.shape[1] * 256 / img.shape[0]), 256)
    else:
        cropSize = (256, int(img.shape[0] * 256 / img.shape[1]))
    img = cv2.resize(img, dsize=cropSize)
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    primaryContour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    floodFill = np.zeros((imgGray.shape[0], imgGray.shape[1]), 'uint8')
    cv2.drawContours(floodFill, primaryContour, -1, 1, 15)
    floodFillMask = np.zeros((imgGray.shape[0] + 2, imgGray.shape[1] + 2), 'uint8')
    cv2.floodFill(floodFill, floodFillMask, (int(imgGray.shape[1] / 2), int(imgGray.shape[0] / 2)), 1)
    rectangleValue = findRectangle(floodFill)
    rectangle = [min(rectangleValue[0], rectangleValue[2]), max(rectangleValue[0], rectangleValue[2]), min(rectangleValue[1], rectangleValue[3]), max(rectangleValue[1], rectangleValue[3])]
    cropImg = img[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]]
    cv2.rectangle(floodFill, (min(rectangleValue[1], rectangleValue[3]), min(rectangleValue[0], rectangleValue[2])), (max(rectangleValue[1], rectangleValue[3]), max(rectangleValue[0], rectangleValue[2])), 3, 2)
    return [cropImg, rectangle, cropSize]


def raSpace(img):
    ratio = 1.0
    threshold = 150
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    width = img.shape[0]
    height = img.shape[1]
    ra = np.zeros((width * height, 2))
    for x in range(width):
        for y in range(height):
            r = math.sqrt((width / 2 - x) * (width / 2 - x) + (height / 2 - y) * (height / 2 - y))
            ra[x * height + y, 0] = r
            ra[x * height + y, 1] = min(imgLab[x][y][1], threshold)
    ra[:, 0] /= max(ra[:, 0])
    ra[:, 0] *= ratio
    ra[:, 1] /= max(ra[:, 1])
    return ra


def processImage(img):
    imgShape = img.shape
    [img, rectangle, cropSize] = findCircle(img)
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    width = img.shape[0]
    height = img.shape[1]
    ra = raSpace(imgLab)
    channel = np.reshape(ra[:, 1], (width, height))
    g = mixture.GaussianMixture(n_components=2, covariance_type='diag', random_state=0, init_params='kmeans')
    imgArraySample = shuffle(ra, random_state=0)[:1000]
    g.fit(imgArraySample)
    label = g.predict(ra)
    label += 1
    label2D = np.reshape(label, (width, height))
    ggLabelRegion = measure.regionprops(label2D, intensity_image = channel)
    ggIntensity = [prop.mean_intensity for prop in ggLabelRegion]
    cervixCluster = ggIntensity.index(max(ggIntensity)) + 1
    mask = np.zeros((width * height, 1), 'uint8')
    mask[label == cervixCluster] = 255
    mask2D = np.reshape(mask, (width, height))
    ccLabels = measure.label(mask2D, background=0)
    region = measure.regionprops(ccLabels)
    area = [prop.area for prop in region]
    regionLabel = [prop.label for prop in region]
    largestCCLabel = regionLabel[area.index(max(area))]
    maskLargestCC = np.zeros((width, height), 'uint8')
    maskLargestCC[ccLabels == largestCCLabel] = 255
    imgMasked = img.copy()
    imgMasked[maskLargestCC == 0] = (0, 0, 0)
    imgMaskedGray = cv2.cvtColor(imgMasked, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(imgMaskedGray, 0, 255, 0)
    kernel = np.ones((11, 11), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=1)
    threshold = cv2.erode(threshold, kernel, iterations=2)
    _, contoursMask, _ = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mainContour = sorted(contoursMask, key=cv2.contourArea, reverse=True)[0]
    i, j, w, h = cv2.boundingRect(mainContour)
    cv2.rectangle(img, (i, j), (i + w, j + h), 255, 2)
    returnImg = img[j:j+h, i:i+w]
    return returnImg


def main():
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
    explorePath = "/Users/praneet/Downloads/train/"
    train_data = []
    train_labels = []
    fileCount = 1
    for root, dirs, files in os.walk(explorePath):
        for fileName in files:
            if fileName.endswith(".jpg"):
                if 'Type_1' in root.split('/')[-1]:
                    train_labels.append(0)
                if 'Type_2' in root.split('/')[-1]:
                    train_labels.append(1)
                if 'Type_3' in root.split('/')[-1]:
                    train_labels.append(2)
                print(fileName)
                filePath = os.path.abspath(os.path.join(root, fileName))
                img = cv2.imread(filePath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = processImage(img)
                fixedSize = (256, 256)
                img = cv2.resize(img, dsize=fixedSize)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                tmp_data = np.zeros((256, 256), dtype=np.float32)
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        tmp_data[x][y] = img[x][y] / 255.0
                train_data.append(tmp_data)
                with open('train_data', 'ab') as fp:
                    np.savetxt(fp, tmp_data)
                fileCount += 1
    print('Total Files: ', fileCount)
    # Evaluation
    print('Evaluation: ')
    evalPath = "/Users/praneet/Downloads/eval/"
    eval_data = []
    eval_labels = []
    fileCount = 1
    for root, dirs, files in os.walk(evalPath):
        for fileName in files:
            if fileName.endswith(".jpg"):
                if 'Type_1' in root.split('/')[-1]:
                    eval_labels.append(0)
                if 'Type_2' in root.split('/')[-1]:
                    eval_labels.append(1)
                if 'Type_3' in root.split('/')[-1]:
                    eval_labels.append(2)
                print(fileName)
                filePath = os.path.abspath(os.path.join(root, fileName))
                img = cv2.imread(filePath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = processImage(img)
                fixedSize = (256, 256)
                img = cv2.resize(img, dsize=fixedSize)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                tmp_data = np.zeros((256, 256), dtype=np.float32)
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        tmp_data[x][y] = img[x][y] / 255.0
                eval_data.append(tmp_data)
                with open('eval_data', 'ab') as fp:
                    np.savetxt(fp, tmp_data)
                fileCount += 1
    with open('eval_labels', 'ab') as fp:
        pickle.dump(eval_labels, fp)
    with open('train_labels', 'ab') as fp:
        pickle.dump(train_labels, fp)

    newchk = []
    for x in train_labels:
        newchk.append(x)

    with open('train_labels', 'rb') as fp:
        itemlist = pickle.load(fp)

    newchklst = []
    for x in itemlist:
        newchklst.append(x)

    print('Length: ', len(newchk), len(newchklst))
    newchk = np.array(newchk)
    newchklst = np.array(newchklst)
    tempo = newchk - newchklst
    counter = 0
    for x in tempo:
        if x == 0:
            counter += 1
        else:
            print(x)
    print('Count: ', counter)

    content = np.loadtxt('train_data')
    print(len(content))

    test = []
    temp = 0
    tmp = []
    for x in content:
        if temp == 255:
            tmp.append(x)
            test.append(tmp)
            temp = 0
            tmp = []
        else:
            tmp.append(x)
            temp += 1

    print(len(test))

    chk = []
    for x in test:
        for y in x:
            for z in y:
                chk.append(z)
    #
    nchk = []
    for x in train_data:
        for y in x:
            for z in y:
                nchk.append(z)
    #
    print(len(chk), len(nchk))
    cnt = 0
    for x in range(len(chk)):
        val = chk[x] - nchk[x]
        if val == 0:
            cnt += 1
        else:
            print(val)
    print(cnt)

    test = np.array(test)
    # print(test)
    print(len(test), len(train_data))

    count = 0
    lol = train_data - test
    for x in lol:
        for y in x:
            for z in y:
                if z == 0:
                    count+=1
                else:
                    print(z)
    print(count)

    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)


if __name__ == '__main__':
    main()
