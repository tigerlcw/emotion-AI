# 프로젝트에 필요한 라이브러리 셋팅
# Tensorflow / keras API 활용
import random
import copy  # 복사 라이브러리
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
keyfacial_df_copy = copy.copy(keyfacial_df)

columns = keyfacial_df_copy.columns[:-1]  # 이미지를 제외한 모든 데이터 복사
print(columns)  # index 확인

# 이미지 증강 작업 - [1] 좌우 반전
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(  # 복사 한 후 keyfacial_df_copy의 'Image' 열에 저장
    lambda x: np.flip(x, axis=1))  # 람다 함수 적용해 x 입력 값에 실제 픽셀 값을 넣고 axis=1 을 이용해 y 축 뒤집기

# y축 값은 같으나 x축 값이 다르므로 x축 값만 선택 가능하게 반복문 작성
for i in range(len(columns)):
    if i % 2 == 0:
        keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(
            lambda x: 96. - float(x))  # 데이터 포인트 뒤집기

# 좌우 반전 이미지 확인 (원본 이미지)
plt.imshow(keyfacial_df['Image'][0], cmap='gray')  # 0번째 행
for j in range(1, 31, 2):
    plt.plot(keyfacial_df.loc[0][j-1], keyfacial_df.loc[0][j], 'rx')

# 좌우 반전 이미지 확인 (변경된 이미지)
plt.imshow(keyfacial_df_copy['Image'][0], cmap='gray')  # 0번째 행
for j in range(1, 31, 2):
    plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')

# 좌우 반전 데이터 프레임 생성
augmented_df = np.concatenate(
    (keyfacial_df, keyfacial_df_copy))  # concatenate 배열 연결

print(augmented_df.shape)  # 복사된 결과 확인 2140 + 2140 = 4280

# 이미지 증강 작업 - [2] 밝기 변경
# x의 픽셀 값을 가져온 후 픽셀 값을 곱한다. 곱하면 이미지의 밝기가 높아진다.
# 픽셀 범위는 0~255 사이로 해야한다.
keyfacial_df_copy = copy.copy(keyfacial_df)
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(
    lambda x: np.clip(random.uniform(1.5, 2) * x, 0.0, 255.0))  # 1.5부터 2 사이의 랜덤 숫자를 고르고 곱한다.
augmented_df = np.concatenate((augmented_df, keyfacial_df_copy))
print(augmented_df.shape)  # 복사된 결과 확인 2140 + 2140 + 2140 = 6420

# 이미지 밝기 증가 (확인)
plt.imshow(keyfacial_df_copy['Image'][0], cmap='gray')
for j in range(1, 31, 2):
    plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')

# 데이터 정규화 및 훈련 데이터 준비 수행
img = augmented_df[:, 30]  # 모든 이미지 가져오기
img = img/255.  # 가져온 다음 이미지 정규화 작업

# 비어있는 배열 생성 shape (x, 96, 96, 1)
X = np.empty((len(img), 96, 96, 1))

# 가지고 있는 모든 데이터 크기를 96*96로 확장 (배치 포맷 형태로)
for i in range(len(img)):
    X[i, ] = np.expand_dims(img[i], axis=2)

# array type을 float32로 변경
X = np.asarray(X).astype(np.float32)
print(X.shape)

# y 좌표 작업
y = augmented_df[:, :30]
y = np.asarray(y).astype(np.float32)
print(y.shape)

# 데이터를 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 20% 테스트 / 80% 학습


# 주요 얼굴 포인트 탐지를 위한 심층 잔차 신경망 구축
# 잔차 블럭 정의
def res_block(X, filter, stage):
    # Convolutional_block
    X_copy = X

    f1, f2, f3 = filter

    # Main Path
    # 합성 곱 이용
    X = Conv2D(f1, (1, 1), strides=(1, 1), name='res_'+str(stage) +
               '_conv_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPool2D((2, 2))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_conv_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name='res_' +
               str(stage)+'_conv_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_conv_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='res_' +
               str(stage)+'_conv_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_conv_c')(X)

    # Short path
    X_copy = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='res_' +
                    str(stage)+'_conv_copy', kernel_initializer=glorot_uniform(seed=0))(X_copy)
    X_copy = MaxPool2D((2, 2))(X_copy)
    X_copy = BatchNormalization(
        axis=3, name='bn_'+str(stage)+'_conv_copy')(X_copy)

    # ADD
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    # Identity Block 1
    X_copy = X

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name='res_'+str(stage) +
               '_identity_1_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_1_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name='res_' +
               str(stage)+'_identity_1_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_1_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='res_'+str(stage) +
               '_identity_1_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_1_c')(X)
    # Short path에서는 그냥 Identity mapping만 하면 된다.

    # ADD (X, X_copy 합침)
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    # Identity Block 2
    X_copy = X

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name='res_'+str(stage) +
               '_identity_2_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_2_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name='res_' +
               str(stage)+'_identity_2_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_2_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='res_'+str(stage) +
               '_identity_2_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_2_c')(X)

    # ADD
    X = Add()([X, X_copy])
    X = Activation('relu')(X)
    # 둘을 합한 다음 활성화 함수에 적용하면, 기본적으로 잔차 블록을 구축한다.

    return X


input_shape = (96, 96, 1)

# Input tensorflow shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - stage
X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
           kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3, name='bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)

# 2 - stage
X = res_block(X, filter=[64, 64, 256], stage=2)

# 3 - stage
X = res_block(X, filter=[128, 128, 512], stage=3)


# Average Pooling
X = AveragePooling2D((2, 2), name='Averagea_Pooling')(X)

# Final layer
# Dense 인공 신경망 구축
# 1. 특성 맵들을 펼친다.
X = Flatten()(X)
# 2. Dense()
X = Dense(4096, activation='relu')(X)
# 3. Dropout() -> 네트워크 일반화 성능에 도움을 준다.
X = Dropout(0.2)(X)  # 뉴런 20% 드롭아웃
# 4. Dense()
X = Dense(2048, activation='relu')(X)
# 5. Dropout()
X = Dropout(0.1)(X)  # 뉴런 10% 드롭아웃
# 6. Dense()
X = Dense(30, activation='relu')(X)

# 첫 번째 나만의 모델 생성
model_1_facialKeyPoints = Model(inputs=X_input, outputs=X)
model_1_facialKeyPoints.summary()  # 모델 요약본 출력

# 훈련된 주요 얼굴 포인트 감지 모델 성능 평가
with open('data/detection.json', 'r') as json_file:  # 모델 오픈 detection.json
    json_savedModel = json_file.read()

# 모델 인식
model_1_facialKeyPoints = tf.keras.models.model_from_json(json_savedModel)
model_1_facialKeyPoints.load_weights(
    'data/weights_keypoint.hdf5')  # 가중치 저장된 파일

# Adam 옵티마이저 사용 해당 프로젝트에 효과적으로 최적화를 할 수 있어 사용
# adam 옵티마이저 학습률 0.001 설정 후 모델 컴파일
adam = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_1_facialKeyPoints.compile(
    loss="mean_squared_error", optimizer=adam, metrics=['accuracy'])
