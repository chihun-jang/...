
import numpy as numpy
import tensorflow as tf 
# 아래 두개는 설치를 해줘야함
import tensorflow_hub as hub 
import tensorflow_datasets as tfds 

#아래는 정보를 확인하는 부분
# print(tf.__version__)
# print(tf.executing_eagerly()) #eager은 즉시실행 모드이다.
# print(hub.__version__)
# print("사용가능" if tf.   config.experimental.list_physical_devices("GPU") else "사용 불가능" )


#훈련 data를 6:4로 나누어 사용한다
#15000개는 훈련, 검증에 1만개 사용하고 해당 영화 review는 총 5만개의 dataset
# train_validation_split = tfds.Split.TRAIN.subsplit([6,4])  사라진 API

#아래는 새로운 subsplit API를 사용한 방법이다.
# ds1, ds2 = tfds.load('imdb_reviews', split=[ 'train[:60%]', 'train[60%:]'])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(('train[:60%]', 'train[60%:]'),tfds.Split.TEST),
    as_supervised=True,
)

# display_tfdataset = tfds.list_builders()
#print(display_tfdataset)

#특히 우리가 여기서 확인하는 dataset은 전처리된 정수배열이고, 0은 부정적리뷰, 1은 긍정적 리뷰를 나타낸다.
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

print("트레이닝셋 batch",  train_examples_batch)

print("트레이닝셋 label",  train_labels_batch)


print("*"*100)
#신경망을 만들기 위해 층을 쌓는데 세가지가 고려된다.
#텍스트의 표현을 어떻게 할것인가
#모델에서 몇개의 층을 사용할 것인가.
#각 층에서 얼마나 많은 은닉유닛(hidden unit)을 사용할 것인가

#우리의 입력데이터는 문장이고 예측한 레이블은 0과1이다.
#텍스트를 표현하는 방법중 하나는 임베딩 벡터로 바꾸는것이다.

#우리는 첫번째 층으로 pre trained된 텍스트 임베딩을 사용할수 있는데
#1. 텍스트 전처리에 대한 신경을 쓸 필요가 없다.
#2. 전이학습(Transfet learning)의 장점을 이용
#3. 임베딩은 고정크기라 처리과정이 단순해진다.

#해당 예제에서는 사전훈련된 테스트 임베딩 모델중 한개 사용(세개를 더 사용할수 있다.)
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
# embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

#임베딩을 시키기 위해 tf hub model 을 사용하는 keras layer를 만들어보자.
#몇개의 샘플을 입력하여 테스트를 하는데 입력테스트 길이 상관없이
#출력의 크기는 (num_examples, embedding_dimension)이 돤다
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable = True)

print(hub_layer(train_examples_batch[:3]))

#모델을 만들어보자
model = tf.keras.Sequential()
model.add(hub_layer) #사전에 훈련된 모델을 사용해 입력한 문장을 임베딩 벡터에 매핑,
                     #문장은 토큰으로 나뉘고 토큰의 임베딩을 연결해 반환

model.add(tf.keras.layers.Dense(16, activation='relu')) #위의 벡터는 16개의 은닉유닛을 가진 Dense(완전연결층)으로 주입된다.
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #하나의 출력노드를 가진 완전연결층이다. sigmoid를 사용해서 확률 또는 신뢰도를 표현하는 0~1사이의 실수출력

print(model.summary()) #모델의 모습을 보여주는 부분


print("@"*100)

#모델을 컴파일 하는 부분
#모델이 훈련을 하기위해서는 loss function과 optimizer가 필요하다.
#해당 예제는 binary classification이므로 binary_crossentropy(확률 다루는데 적합)를 사용한다.
model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])


#512개의 샘플로 이무어진 배치에서 20번의 epoch으로 훈련
# 10000개의 검증세트에서 손실과 정확도 모니터링
history = model.fit(train_data.shuffle(10000).batch(512), 
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

print("======================================================")
#두개의 값이 반환되는데 손실과 정확도를 반환한다.
results = model.evaluate(test_data.batch(512),verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" %(name,value))

#해당 예제는 단순한것을 사용해서 87%가 나오지만 고급방법을 사용하면 95%까지 올릴수 잇다.