# utils.py
import os
import glob
import random
from pathlib import PurePath

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from imgaug import augmenters as iaa


# =========================
# STEP 1 - INITIALIZE DATA
# =========================
def getName(filePath: str) -> str:
    """
    รับ path จาก CSV แล้วคืนค่า 'โฟลเดอร์สุดท้าย/ชื่อไฟล์'
    รองรับทั้ง Windows ('\\') และ POSIX ('/')
    """
    p = PurePath(str(filePath))
    parts = p.parts[-2:]  # เช่น ('IMG0', 'image.jpg')
    return str(PurePath(*parts))


def _read_csv_robust(csv_path: str, names):
    """
    อ่าน CSV ให้ทนทานกับ delimiter/บรรทัดเสีย:
    - ให้ pandas เดา delimiter
    - ข้ามบรรทัดที่คอลัมน์ไม่ครบ
    รองรับทั้ง pandas รุ่นเก่า/ใหม่
    """
    kwargs = dict(header=None, names=names, sep=None, engine="python")
    try:
        # pandas >= 1.3
        return pd.read_csv(csv_path, on_bad_lines="skip", **kwargs)
    except TypeError:
        # pandas <= 1.2 (ไม่มี on_bad_lines)
        return pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=False, **kwargs)


def importDataInfo(path: str) -> pd.DataFrame:
    """
    รวมข้อมูลจากทุกไฟล์ log_*.csv ภายใต้โฟลเดอร์ 'path'
    คืน DataFrame คอลัมน์ ['Center','Steering']
    """
    path = os.path.abspath(path)
    columns = ['Center', 'Steering']

    csvs = sorted(glob.glob(os.path.join(path, "log_*.csv")))
    if not csvs:
        raise FileNotFoundError(f"ไม่พบไฟล์ CSV ที่เป็น 'log_*.csv' ใน: {path}")

    frames = []
    for f in csvs:
        df = _read_csv_robust(f, names=columns)
        # ทำให้ path ของภาพอยู่ในรูป 'IMGx/filename.jpg'
        df['Center'] = df['Center'].astype(str).apply(getName)
        # แปลง Steering -> float และทิ้งที่ไม่ใช่ตัวเลข
        df['Steering'] = pd.to_numeric(df['Steering'], errors='coerce')
        df = df.dropna(subset=['Steering'])
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    print('Total Images Imported', data.shape[0])
    return data


# ======================================
# STEP 2 - VISUALIZE AND BALANCE DATA
# ======================================
def balanceData(data: pd.DataFrame, display: bool = True) -> pd.DataFrame:
    nBin = 31
    samplesPerBin = 300
    hist, bins = np.histogram(data['Steering'], nBin)

    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()

    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'].iloc[i] >= bins[j] and data['Steering'].iloc[i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], (nBin))
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()

    return data


# ====================================
# STEP 3 - PREPARE FOR PROCESSING
# ====================================
def loadData(path: str, data: pd.DataFrame):
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        # สร้าง absolute path ของภาพภายใต้โฟลเดอร์ path
        imagesPath.append(os.path.join(path, indexed_data['Center']))
        steering.append(float(indexed_data['Steering']))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering


# ==========================
# STEP 5 - AUGMENT DATA
# ==========================
def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


# ===================
# STEP 6 - PREPROCESS
# ===================
def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


# =====================
# STEP 7 - CREATE MODEL
# =====================
def createModel():
    model = Sequential()
    # หมายเหตุ: Convolution2D(filters, kernel, strides, ...)
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model


# =====================
# STEP 8 - TRAINING GEN
# =====================
def dataGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for _ in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))
