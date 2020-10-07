import tensorflow as tf 

from tensorflow import keras

import numpy as numpy

print(tf.__version__)
# 인터넷 무비 데이터 베이스 dataset을 tf로 부터 가져온다.
imdb = keras.datasets.imdb
# num_words를 지정해준것은  훈련데이터에서 많이 등장하는 상위 1만개 단어 선택
(train_data, train_labels) , (test_data, test_labels)  = imdb.load_data(num_words=10000)

print(f'훈련샘플:{len(train_data)},레이블:{ len(train_labels)}')

#리뷰텍스트는 각 단어를 특정 정수에다가 매칭시켜놓았다.
print(train_data[0])

print(len(train_data[0]),len(train_data[1]) )

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}

#이처럼 처음 몇개의 index는 우리가 사전에 정의를 해주자.
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#위에는 key값이 단어가 되고 value값이 정수 숫자인데
#아래는 key와 value의 순서를 바꾸어 key값이 숫자이고 value를 단어로 해주어 dict 안에서 찾기 쉽게해줬다.
reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])


#아래는 위의 처리내용을 바탕으로 헬퍼함수를 만들어준것.
def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])

#헬퍼함수를 이용해서 문장을 생성한 모양
print(decode_review(train_data[0]))

#리뷰-정수배열을 신경망에 넣기전 tensor로 변환해야하는데 
# 1. one-hot 인코딩으로 0과 1로 이루어진 벡터로 변환하는 방법이있는데
# [3,5]의 단어로 구성된 문장의 경우 3과 5만 1이 되고 나머지가 모두 0인 배열로 변경이가능하다.
# 그런데 이렇게 되면 num_words * num_reviews(여기서는 1만)만큼이 행렬이 필요하므로 memory를 많이 사용한다.

# 다른 방법으로는 정수배열의 길이가 같도록 padding을 추가해서 max_length* num_review크기의 정수 tensor를 만드는건데
#이런 형태의 텐서를 다룰수 잇는 embedding층을 신경망의 첫번째로 사용하면 된다.



#리뷰의 길이를 맞추기 위해서 pad_sequences함수를 사용해서 길이를 통일시켜주자

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding="post",
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value= word_index["<PAD>"],
                                                        padding="post",
                                                        maxlen=256)
print("길이를 맞춘 두개의 data", len(train_data[0]), len(train_data[1]))

print("패딩된 data",train_data[0])


#모델구성

#모델에서 얼마나 많은층을 사용할것인가
# 각층에서 얼마나 많은 은닉유닛을 사용할 것인가  이 두개를 고려하면서 만들어준다.


vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size , 16 , input_shape=(None,))) #정수로 인코딩된 단어를 입력받고 인덱스에 해당하는 임베딩 벡터를 찾는다. 벡터는 모델이 훈련되며 학습된다.
                                                                        #벡터는 출력배열에 새로운 차원으로 추가된다.(batch, sequence, embedding)이런식으로
model.add(keras.layers.GlobalAveragePooling1D()) #sequence 차원에 대해 평균을 계산해 샘플에 대해 고정된 길이의 출력벡터반환. 
model.add(keras.layers.Dense(16,activation="relu")) #위에서 내려온 고정길이의 출력벡터는 16개의 은닉유닛을 가진 fully-connected (Dense)를 거친다.
model.add(keras.layers.Dense(1, activation = "sigmoid")) # 하나의 출력노드를 가진 완전 연결층이고 0~1사이의 실수를 출력한다.

print(model.summary())


#은닉유닛이란
#출력의 개수는 각 층이가진 표현공간의 차원이 된다. 즉 내부표현을 학습할때 허용되는 network 자유도의 양이다.
# 모델에 많은 은닉유닛(고차원의 표현공간)과 층이 있다면 network는 더 복잡한 표현을 학습할수 있다.
# 하지만 비용이 많이들고 원치않는 학습을 할수도 있다. 
# 이런 경우에는 훈련데이터의 성능은 향상되지만 test data에서는 그러지 못하고 일ㄹ overfitting이라고 한다.

#모델 컴파일하기
model.compile(optimizer='adam',
            loss="binary_crossentropy",
            metrics=['accuracy'])

#검증할 세트 만들기
#모델은 만난적 없는 data에서 정확도를 확인하는게 좋다. 
#test data이전에 test를 해보는 느낌이다.

x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#모델 훈련시키는 부분
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data = (x_val, y_val),
                    verbose = 1)

print("히스토리 출력", history)

#모델 평가하는 부분
results = model.evaluate(test_data, test_labels, verbose =2)
print(results)

#model fit 을 통해 반환 되는 객체로 모니터링 해보기
history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(acc) + 1)

#bo 는 파란색 점
plt.plot(epochs, loss, 'bo', label = "Training loss")

#b 는 파란실선
plt.plot(epochs , val_loss , 'b' , label="Validation loss")

plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend()

plt.show()


plt.clf()  # r그림을 초기화 시키는 코드

plt.plot(epochs, acc,'bo', label="Training acc")
plt.plot(epochs, val_acc,'b', label = "Validation acc")

plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.show()


# 이렇게 그래프로 보면 epoch 20전후에서 validation의 acc가 더이상 늘어나지 않는게 보이는데
# overfitting때문이다. 이때부터는 과도하게 training data에 최적화 되어가서 일반화하기 어려워진다.
# 따라서 overfitting을 막기위해 20번째 epochs근처에서 훈련을 멈출수잇는데 callback을 사용하여 자동으로 멈추는 방법을 배울꺼임.