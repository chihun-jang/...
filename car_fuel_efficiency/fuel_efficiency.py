import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns #산점도 행렬을 그리기 위해 사용함

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#dataset download받는 부분
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

print(dataset_path)

#Pandas 이용하요 data읽어들이기
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year', 'Origin']

#pd 로 부터 data read 해오고 raw_dataset에 넣어놓는데
raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values="?",
                            comment="\t",sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()

#마지막(tail) 5개만 보여준다.
dataset.tail()

print(dataset.tail())

#dataset중 na가 있는 col 의 sum
print(dataset.isna().sum())

dataset= dataset.dropna()

#origin의 data들은 수치를 나타내는 것이 아니라 범주를 나타내는 data이므로
#one-hot encoding으로 변환해준다.
origin = dataset.pop('Origin')

#그리고 dataset에 각 나라에 해당하는 col을 추가해주고 origin이 나타내는 값에따라 표시를해준다.
dataset['USA'] = (origin == 1)* 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

print(dataset.tail())

# 데이터셋을 훈련세트와 테스트 세트로 분할하기.
train_dataset = dataset.sample(frac=0.8, random_state=0)

#인덱스를 드랍
test_dataset = dataset.drop(train_dataset.index)

#train_set에서 몇개의 열을 선택해 seaborn을 만들어보자
# sns.pairplot(train_dataset[["MPG", "Cylinders","Displacement","Weight"]],diag_kind = "kde")
# plt.show()

#통계 확인하기
train_stats = train_dataset.describe()
train_stats.pop("MPG")
#dataset을 전치한다.
train_stats = train_stats.transpose()
print(train_stats)

#특성과 레이블 분리하기
#특성에서 레이블 분리하고, 이를 예측해볼 것이다.

train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

#데이터 정규화 시키기
#특성의 스케일과 범위가 다르면 normalization을 하는것이 권장된다.
#정규화 하지않으면 훈련시키기가 힘들고, 입력에 의존적인 모델이 만들어진다.

#의도적으로 훈련세트만 사용해서 통계치를 생성했다. 이는 testset을 정규화할때도 사용하는데 훈련에 사용한것과 동일한 분포를 투영하기 위함.

def norm(x):
    return (x-train_stats['mean'])  / train_stats['std'] #평균과 표준편차

#이때 통계치는 모델에 주입되는 모든 data에 적용되어야 한다.

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#모델만들기

# 두개의 dense 은닉층으로 Sequential 모델을 만들자
# 출력층은 하나의 연속값을 반환

#재사용하기 위해 model함수를 생성했다.
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation = 'relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64,activation="relu"),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=['mae','mse'])

    return model

model = build_model()
#모델의 구조를 봐봅시다
print(model.summary())

#모델을 실행하기 위해 trainset에서 10샘플을 하나의 배치로 만들어서 넣어주자
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

#모델의 훈련과정
#그리고 진행과정을 확인할수 잇께 epoch이 끝날때마다 점을 찍고 100단위로 출력해주자
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch %100 == 0:
            print(epoch,"번했습니다")
        print(".",end="")

EPOCHS = 1000

# history = model.fit(
#     normed_train_data, train_labels,
#     epochs= EPOCHS, validation_split = 0.2, verbose=0,
#     callbacks=[PrintDot()]
# )

# #history에 저장된 통계치를 시각화해보자
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())

import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'],hist['mae'],label="Train Error")
    plt.plot(hist['epoch'], hist['val_mae'], label= "Val Error")
    plt.ylim([0,5])
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel("Mean Square Error [$MPG^2$]")
    plt.plot(hist['epoch'], hist['mse'],label="Train Error")
    plt.plot(hist['epoch'], hist['val_mse'],label="Val Error")
    plt.ylim([0,20])
    plt.legend()
    plt.show()


# plot_history(history)

#실행값을 보면 특정시점을 기준으로 모델의 성능이 향상되지않는 것을 확인할수 잇는데
#모델의 훈련과정을 수정하여 성능이 향상되지않으면 훈련을 멈추도록 만들어보자
#epoch 마다 훈련상태를 점검하기 위해 EarlyStopping callBack을 사용하자.

#patiencs 매개변수는 성능향상을 위해 체크할 epoch횟수이다.

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose = 0, callbacks=[early_stop,PrintDot()])

# plot_history(history)


#테스트 세트로 모델의 성능을 확인해보자

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose = 2)

print(f"테스트 세트의 평균 절대 오차:{mae:5.2f} MPG")

#테스트 세트의 샘플을 사용해 MPG값을 예측해보자

test_predictions = model.predict(normed_test_data).flatten()

#그리고 예측한 것을 실제값과 비교하여 시각화해보자
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100,100], [-100,100])
plt.show()

#다음으로는 오차의 분포도 알아보자

plt.clf()
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel('Count')
plt.show()

#가우시안 분포가 아니지만 테스트 샘플수가 작아서 그럴 것

#정리
#MSE :<평균제곱의 오차>는 regression 에서 자주 사용하는 cost함수이다.(classify에서 사용하는것과 다르다)
#MAE : <평균 절댓값 오차>는 regression에서 사용되는 평가지표다.
#입력 데이터의 범위가 다양할때 동일범위와 스케일을 가지도록 normalization을 해줘야한다.
#train_data가 많지않다면 overfitting을 피하기위해 dense의 갯수가 적은 소규모 network을 선택하는게 좋다
#Early Stopping은 overfitting을 방지하기 위한 좋은 방법