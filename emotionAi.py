# 프로젝트에 필요한 라이브러리 셋팅
# Tensorflow / keras API 활용
import pandas as pd
import numpy as np
import os
import PIL  # 이미지 처리 도움 라이브러리
import seaborn as sns  # 데이터 시각화
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 데이터 셋 로드
keyfacial_df = pd.read_csv('data/data.csv')
# 해당 데이터 셋에는 왼쪽 눈, 오른쪽 눈, 코, 윗입술, 아랫입술 등 15개의 인식 포인트가 있다.
print(keyfacial_df)
# 총 2140개의 이미지, 2140개의 데이터 포인트와 31개의 열이 있다.

print(keyfacial_df.info())
# 이미지를 제외하고 모든 데이터 타입은 float64로 되어있다.

print(keyfacial_df.isnull().sum())
# null 값 체크

# 람다 함수를 이용 데이터를 reshape 해준다. 처리가 편하기 위해 96*96 픽셀 형태로 바꿔준다.
# 처리할 때는 'Image' 열을 이용
keyfacial_df['Image'] = keyfacial_df['Image'].apply(
    lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96, 96))

# reshape 확인
print(keyfacial_df['Image'][0].shape)  # 0번째 행을 가져와 96 * 96 변경되었는지 확인

# 이미지 시각화 작업
# 3행 3열 총 9개의 랜덤 이미지 출력
fig = plt.figure(figsize=(20, 20))  # figure 사이즈 설정

for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    i = np.random.randint(1, len(keyfacial_df))
    plt.imshow(keyfacial_df['Image'][i], cmap='gray')  # 컬러는 회색으로 지정
    for j in range(1, 31, 2):
        plt.plot(keyfacial_df.loc[i][j-1],
                 keyfacial_df.loc[i][j], 'rx')  # 얼굴 이미지의 행 지정 후 필요한 정보 출력
        # 주요 얼굴 포인트 표시 후 시각화 출력

# 이미지 증강 작업
# 현재 가지고 있는 입력 데이터에 의존하지 않기 위해 추가적인 데이터 세트를 생성한다.
