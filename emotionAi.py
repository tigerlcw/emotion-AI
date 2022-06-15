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
