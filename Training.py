# train.py
print('Setting UP')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ลด log ของ TF

import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# เปลี่ยน utlis -> utils ให้ตรงชื่อไฟล์ที่แก้แล้ว
from utils import (
    importDataInfo, balanceData, loadData,
    createModel, dataGen
)

# (ทางเลือก) เปิด memory growth ถ้ามี GPU ป้องกัน OOM ตอน init
try:
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# ===== STEP 1 - INITIALIZE DATA =====
# ใช้ absolute path ให้ชัดเจนขึ้น
DATA_DIR = os.path.abspath('DataCollected')
data = importDataInfo(DATA_DIR)
print(data.head())

# ===== STEP 2 - VISUALIZE AND BALANCE DATA =====
data = balanceData(data, display=True)

# ===== STEP 3 - PREPARE FOR PROCESSING =====
imagesPath, steerings = loadData(DATA_DIR, data)

# ===== STEP 4 - SPLIT FOR TRAINING AND VALIDATION =====
xTrain, xVal, yTrain, yVal = train_test_split(
    imagesPath, steerings,
    test_size=0.2, random_state=10, shuffle=True
)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

# ===== STEP 5/6 handled in generator & utils =====

# ===== STEP 7 - CREATE MODEL =====
model = createModel()
model.summary()

# ===== STEP 8 - TRAINING =====
BATCH_TRAIN = 100
BATCH_VAL = 50

# ป้องกันหารลงศูนย์ กรณีรูปน้อยมาก
#steps_per_epoch = max(1, len(xTrain) // BATCH_TRAIN)
steps_per_epoch    = 30
validation_steps = max(1, len(xVal) // BATCH_VAL)

history = model.fit(
    dataGen(xTrain, yTrain, BATCH_TRAIN, trainFlag=1),
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=dataGen(xVal, yVal, BATCH_VAL, trainFlag=0),
    validation_steps=validation_steps,
    verbose=1
)

# ===== STEP 9 - SAVE THE MODEL =====
model.save('model.h5')
print('Model Saved -> model.h5')

# ===== STEP 10 - PLOT THE RESULTS =====
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
