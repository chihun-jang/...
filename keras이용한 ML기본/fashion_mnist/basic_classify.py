import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images , train_labels) , (test_images, test_labels) = fashion_mnist.load_data()

dress_name = ['T-shirt/top', 'Trouser' , 'Pullover','Dress','Coat','Snada','Shirt','Sneaker','Bag','Ankle-boot']

print('학습이미지 구조',train_images.shape)
print('학습이미지 양',len(train_labels))

print('라벨구조',type(train_labels))

print('테스트이미지 구조',test_images.shape)
print('테스트이미지 양',len(test_labels))


## 데이터 전처리 (네트워크를 훈련시키기 전에 데이터를 전처리해야한다. )

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#픽셀 하나하나가 나타내는 범위가 0~255까지의 범위인데 이를 모델에 넣기전에 0~1 사이로 조정한다. 
train_images = train_images/ 255.0
test_images = test_images/ 255.0


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(dress_name[train_labels[i]])
# plt.show()



#모델 구성
# 신경망 모델을 만들기위해서 layer를 구성한다음 모델을 컴파일 합니다
# 신경망의 기본 구성요소는 층이다. 층은 data에서 피쳐를 추출하고 딥러닝은 이러한 층의 연결로 구성된다.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  #2차원 배열 28*28을 784픽셀의 1차원 배열로 변환
    keras.layers.Dense(128, activation = 'relu'), #Dense는 밀집연결 또는 완전연결이라고 한다. 128개의 노드(뉴런)를 가지고
    keras.layers.Dense(10, activation = 'softmax'), # 10개 노드의 softmax레이어, 10개의 확률을 반환한다(각 노드는 이미지가 각 노드일 확률)
])

#모델 컴파일(모델 훈련전 필요단계)
#손실함수(loss func)  : 훈련하는 동안 모델의 오차를 측정하고, 모델이 올바른 방향으로 학습할수 있게 최소화한다.(cost func)
#옵티마이저(optimizer) : data와 loss func를 바탕으로 모델의 업데이트 방향결정
#지표(Metrics) : 훈련단계와 테스트 단계를 모니터링 하기위해 사용, 

model.compile(optimizer = 'adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#모델훈련
#1. train_Data를 모델에 넣는다.
#2. 모델이 img와 label을 매핑하는 방법을 배운다
#3. test data에 대한 모델의 예측을 만들고 test_label과 맞는지 비교한다

model.fit(train_images,train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)
print('테스트 정확도: ', test_acc)


predictions =model.predict(test_images)

print("첫번째 예측: ", predictions[0])

print("예측하는 라벨값: ", np.argmax(predictions[0]))

print("실제 라벨값: ", test_labels[0])

def plot_image(i, predictions_array, true_label,img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color= 'blue'
    else:
        color="red"
    plt.xlabel(f'{dress_name[predicted_label]}     {100*np.max(predictions_array)}%    ({dress_name[true_label]})', color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10),predictions_array,color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')



# i번째 원소의  image와 예측 신뢰도 점수 배열을 확인할수 있다.
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i,predictions, test_labels)
# plt.show()


#아래는 여러개의 fashion image를 테스트해서 plt로 띄워주기 위한 코드입니다.
# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다

# num_rows =5
# num_cols= 3

# num_images = num_rows* num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i,predictions,test_labels)
# plt.show()

img = test_images[0]
print('테스트 이미지 구조',img.shape)


#tf.keras 모델은 한번에 샘플의 묶음, 배치로 예측을 만드는데 최적화 되어있따.
#이떄 하나의 이미지를 사용하더라도 2차원 배열로 만들어야한다.

img2 = (np.expand_dims(img,0))

print(img2.shape)

#하나의 이미지를 가져왔고 그 이미지에 대한 예측을 해보는 것이다.

predictions_single = model.predict(img2)
print(predictions_single)

#그리고 이 이미지에 대한 확률값을 그래프로 보기 위한 코드입니다.
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10),dress_name, rotation=45)
plt.show()
# argmax를 통해서 확률로 바꿔주고 dress_name에서 옷의 이름과 매칭을 시켜준다.
print("예측라벨 : ", np.argmax(predictions_single[0]))
print("예측 드레스: " ,dress_name[np.argmax(predictions_single[0])])