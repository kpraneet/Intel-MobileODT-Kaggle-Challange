import os
import cv2
import math
import numpy as np
import tensorflow as tf
import pandas as pd


from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python import SKCompat


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


def trainTypes(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features, [-1, 256, 256, 1])
    # Convolutional Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Convolutional Layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 3
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    # Convolutional Layer 4
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 4
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    # Convolutional Layer 5
    conv5 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer 5
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool5_flat = tf.reshape(pool5, [-1, 8 * 8 * 512])
    dense = tf.layers.dense(inputs=pool5_flat, units=1024, activation=tf.nn.relu)
    # Dropout
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=3)
    loss = None
    train_op = None
    # Calculate loss for both Train and Eval
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    # Configure Training
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.01,
            optimizer="SGD")
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }
    # Return the object
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_arguments):
    mnist_classifier = SKCompat(learn.Estimator(model_fn=trainTypes, model_dir="/Users/praneet/Downloads/model"))
    print('Predictions: ')
    evallst = []
    outputlst = []
    fileCount = 1
    evaluatePath = "/Users/praneet/Downloads/test/"
    lst = []
    lst.append("image_name")
    lst.append("Type_1")
    lst.append("Type_2")
    lst.append("Type_3")
    evallst.append(lst)
    outputlst.append(lst)
    for root, dirs, files in os.walk(evaluatePath):
        for fileName in files:
            if fileName.endswith(".jpg"):
                eval_data = []
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
                eval_data = np.array(eval_data)
                # Predict values for each image
                predictions = mnist_classifier.predict(x=eval_data)
                print(fileName, predictions)
                lst = []
                lst.append(fileName)
                for x in predictions['probabilities']:
                    for y in x:
                        lst.append(y)
                outputlst.append(lst)
                if lst[2] > 0.5:
                    lst = []
                    lst.append(fileName)
                    lst.append(0)
                    lst.append(1)
                    lst.append(0)
                if lst[3] > 0.5:
                    lst = []
                    lst.append(fileName)
                    lst.append(0)
                    lst.append(0)
                    lst.append(1)
                if lst[1] > 0.5:
                    lst = []
                    lst.append(fileName)
                    lst.append(1)
                    lst.append(0)
                    lst.append(0)
                if lst[1] == 0 or lst[1] == 1:
                    print("Non Ambiguous Prediction")
                evallst.append(lst)
                fileCount += 1
    print('Total files: ', fileCount)
    df = pd.DataFrame(evallst)
    df.to_csv('output_normalized_check.csv', index=False, header=False)
    df = pd.DataFrame(outputlst)
    df.to_csv('output_integers_check.csv', index=False, header=False)


if __name__ == '__main__':
    tf.app.run()
