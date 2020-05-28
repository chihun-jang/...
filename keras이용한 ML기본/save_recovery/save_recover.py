#train 도중이나 train이 끝난후에 model을 저장할수있다.
#그럼 여러번에 끊어서 훈련할수있고, 다른사람에게 공유할수도 있다.
#연구한 모델과 기법을 공개할때는 1. 모델을 만드는 코드 , 2. 모델의 훈련된 가중치 또는 파라미터 를 공유하여 다른사람들이 개선할수도 있다.

# 사용하는 API에 따라 여러 방법으로 tf model을 저장할수있는데, 
#여기서는 tf.keras 를 사용하고 그 이외에는 https://www.tensorflow.org/guide/saved_model?hl=ko
#https://www.tensorflow.org/guide/eager?hl=ko#object-based_saving 두개의 link를 참고하자

import os 

import tensorflow as tf
from tensorflow import keras

#우리는 간단하게 훈련시키기 위해서 MNIST Dataset을 사용하고 가중치를 저장해볼것이다.

(train_images, train_labels) , (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#빨리해보기 위해 앞에 1000개만 가져온다
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1,28*28)/255.0
test_images = test_images[:1000].reshape(-1,28*28)/255.0

#간단한 Seq 모델을 반환한다.
def create_model():
    model = tf.keras.models.Sequential([
        #아래에 보면 input_shape의 크기는 우리가 input으로 받는 img의 28*28 이고 
        #512개의 애들이 784개를 받을수 있으니까 param은 401920개가 나온다.
      keras.layers.Dense(512, activation = "relu", input_shape = (784,)),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(10, activation= "softmax")  
    ])

    model.compile(optimizer="adam", 
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
    return model


model = create_model()
# print(model.summary())


#훈련 중간과 훈련 마지막에 checkpoint를 자동을 저장하는것이 많이 사용하는 방법,
#tf.kears.callbacks.ModelCheckpoint는 이럴떄 사용하는 callback이다.

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,verbose=1)

# model = create_model()

# model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images,test_labels),
# callbacks = [cp_callback]) #훈련할떄 callback을 전달해서 사용한다.





# #훈련하지 않은 새로운 모델을 만들어보자, 가중치를 공유해야할때는 원본과 같은 구조로 만들어야한다.
# #훈련하지 않은 new model을 만들고 testset에서 평가해보자.

# model = create_model()

# loss, acc = model.evaluate(test_images, test_labels, verbose = 2)

# print(f'not trained model acc: {100*acc:5.2f}%')

# #가중치를 로드하고 다시 평가해보자
# model.load_weights(checkpoint_path)
# loss,acc = model.evaluate(test_images, test_labels, verbose =2)
# print(f"recover model acc : {100*acc:5.2f}%")

# #checkpoint callback param

# 파일이름에 epoch 번호를 포함시킨다 ('str.format' format)
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path, save_weights_only=True, verbose=1,
#     #다섯번째 epoch마다 가중치를 저장한다.
#     period = 5)

# model = create_model()

#아래는 모델을 훈련시키고 checkpoint를 저장해놨으므로 다시 쓰지않는것
# model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels),
#           callbacks=[cp_callback],verbose=0)  # 훈련할떄 callback을 전달해서 사용한다.

# latest = tf.train.latest_checkpoint(checkpoint_dir)

# print(latest)

#tf는 기본적으로 최근 5개의 checkpoint만 저장한다.
#model을 초기화하고 checkpoint를 로드해서 test해보자

# model2 = create_model()
# model2.load_weights(latest)

# loss, acc = model2.evaluate(test_images, test_labels, verbose=2)
# print(f"recover된 model의 acc: {100*acc:5.2f}")

#체크포인트 파일: ckpt 포맷의 파일에 저장하는데 훈련된 이진 포맷의 가중치가 저장된다.
# 모델의 가중치를 포함하는 하나이상의 (shard)
# 가중치가 어느 shard에 저장되어있는지 나타내는 index file

#수동으로 가중치 저장하기
#수동으로 가중치를 저장하는것도 쉽다. Model.save_weights 메서드를 사용


#가중치를 저장한다.
# model.save_weights('./checkpoints/my_checkpoint')
# #가중치 복원
# model3 = create_model()
# model3.load_weights('./checkpoint/my_checkpoint')

# loss,acc = model3.evaluate(test_images,  test_labels, verbose=2)
# print(f"복원된 모델의 정확도: {100*acc:5.2f}%")

#model 전체저장하기
# 가중치, model ,optimizer 까지 포함해서 저장할수있는데, 원본코드를 사용하지않고 나중에 동일하게 시작가능
# 전체모델을 젖아하면 Tensorflow.js 로 모델을 load 한다음 web에서 모델을 훈련하고 실행할수 있음(HDF5, Saved Model) 혹은 모바일에 맞게 변환하여 Tf lite 실행가능

# model4 = create_model()

# model4.fit(train_images,train_labels, epochs = 5)
# #전체모델을 HDF5파일로 저장한다.
# model4.save("my_model.h5")


# #파일로 부터 가져와 모델을 다시 만들자

# load_model = keras.models.load_model('my_model.h5')

# # print(load_model.summary())

# loss, acc = load_model.evaluate(test_images, test_labels, verbose=2)
# print(f"load된 model의 acc: {100*acc:5.2f}%")


#가중치의 값, 모델 구조, 옵티마이저 설정 모두 저장한다.
#keras 는 모델의 구조를 호가인하고 저장하는데 현재는 tf.train(텐서플로 옵티마이저)를 저장할수 없다.
#이럴때는 load 후에 compile만 다시해주면 된다.


#save_model 사용해서 모델 저장하기

model5 = create_model()

model5.fit(train_images, train_labels, epochs=5)
#time utils을 불러와서 경로에다가 timestamp로 저장을 해준다.
import time
saved_model_path = f"./saved_models/{int(time.time())}"
#keras에서 지원하는 export_saved_model 얘는 나중에 사라질수도 있다.
tf.keras.experimental.export_saved_model(model5, saved_model_path)
print(saved_model_path)

#그리고 저장된 model을 통째로 불러와서 확인하자.
load_model2 = tf.keras.experimental.load_from_saved_model(saved_model_path)

print(load_model2.summary())

print("test_set의 size ",load_model2.predict(test_images).shape)

#이 모델을 평가하려면 compile을 해야하는데 단지 model의 배포라면 안해도 된다.
load_model2.compile(optimizer=model5.optimizer, loss="sparse_categorical_crossentropy",metrics=['accuracy']) #optiizer를 가져와서 쓰는 것을 체크

#복원된 모델을 평가한다.

loss, acc = load_model2.evaluate(test_images, test_labels, verbose=2)

print(f"recover model acc=============: {100*acc:5.2f}")


#tf.keras 에서는 tf.keras 로 모데릉ㄹ 저장하고 load 하는 정보를 더 볼수있다.
#저수준의 텐서플로의 저장 및 복원도 있다.