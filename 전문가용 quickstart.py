from __future__ import absolute_import, division,print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0 , x_test / 255.0

#채널 차원을 추가한다

x_train = x_train[...,tf.newaxis]
x_test = x_test[...,tf.newaxis]


train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#모델 상속을 통해서 MyModel을 만들어주자
class MyModel(Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1 = Conv2D(32,3,activation = 'relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation = 'relu')
        self.d2 = Dense(10, activation = 'softmax')

    def call(self,x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
    
    
model = MyModel()

# loss함수로  SparseCategoricalCrossentropy를 선택했다.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
# 옵티마이저 모델로 아담을 선택했다.

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

# 모델을 훈련하는 부분
@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

#모델 테스트 부분
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)
    
    for test_images,test_labels in test_ds:
        test_step(test_images, test_labels)


    print(f"에포크: {epoch+1,}, 손실:{train_loss.result()}, 정확도: {train_accuracy.result()*100}, 테스트 손실: {test_loss.result()} , 테스트 정확도: {test_accuracy.result()*100}")
    
    #모델을 테스트 하는 부분


#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# AVX2 와 같은 명령어를 사용하면 보다 더 고급연산을 빠르게 수행할수있는데 이를 물어보는 정도
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
