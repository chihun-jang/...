
#overfitting된 모델은 train set에서는 높은 성능을 보이지만 test set or new set에 대해서는 좋은 성능을 보여주지 못한다.
#반대로 underfitting은 test set의 성능이 향상될 여지가 아직 있을때 일어난다(모델이 너무 단순하거나, 규제가 많거나, 훈련이 너무 빨리 끝난경우)
#즉 network가 train set에서 적절한 pattern을 학습하지 못했다는 말이다.

#따라서 우리는 과대적합과 과소적합 사이에서 균형을 잡아야한다.

#과대적합을 막는 방법 
#1.더많은 훈련데이터사용
#2.규제와 같은 기법 사용(모델이 저장할수 있는 정보의 양과 종류에 제약)// 가중치 규제와 드롭아웃도 규제의 기법중하나

import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt    

print(tf.__version__)

#이번 예제에서는 embedding을 사용하지 않고 multi-hot encoding을 해볼꺼다
#이를 통해 빠르게 과대적합을 시켜보자

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # 0으로 채워진 (len(sequences), dimension)크기의 행렬을 만든다  
    results = np.zeros((len(sequences),dimension))
    for i, word_indices in enumerate(sequences):
        results[i,word_indices] = 1.0   #reslt[i]의 특정인덱스만 1로 설정한다.
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension = NUM_WORDS)

# plt.plot(train_data[0])
# plt.show()

#과제적합을 막는 가장 간단한 방법은 모델의 규모를 축소하는것, 즉 학습가능한 param의 수를 줄이는 건데. param은 model의 layer + unit의 갯수로 결정된다.
#딥러닝에서는 model의 학습가능한 param을 model의 용량이라고 말하기도 한다. 이런 모델의 경우 일반화하지않고도 정확하게 매핑시킬수있겠지만 일반화가 안되므로 효용성이 덜하다.

#우리가 해결해야할 문제는 일반화이다

#반면 너무 용량이 적으면 쉽게학습할수 없고, 더 압축된 표현으로 학습해야한다.
#안타깝게도 적절한 크기와 구조의 모델을 결정하는 방법은 없다

#이러한 모델을 찾는 방법은 적은 용량부터 시작해서 loss가 감소할때까지 새로운 layer와 크기를 늘리는 것이 좋다.

baseline_model = keras.Sequential([
    # .summary method 때문에 input_shape가 필요하다.
    keras.layers.Dense(16,activation = "relu", input_shape=(NUM_WORDS, )),
    keras.layers.Dense(16,activation="relu"),
    keras.layers.Dense(1, activation= "sigmoid")
])

#모델을 컴파일한다.
baseline_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', 'binary_crossentropy'])

print(baseline_model.summary())

#모델을 훈련시킨다.
baseline_history = baseline_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels),verbose=2)

#위의 모델을 학습시켜보면 val_accu가 감소하다가 다시 증가하는 것을 볼수 있다.

#작은 모델
# smaller_model = keras.Sequential([
#     keras.layers.Dense(4, activation = "relu", input_shape= (NUM_WORDS,)),
#     keras.layers.Dense(4, activation="relu"),
#     keras.layers.Dense(1, activation="sigmoid")]
# )

# smaller_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', 'binary_crossentropy'])

# print(smaller_model.summary())

# smaller_history = smaller_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

#큰 모델, 큰모델을 만들면 과대적합이 얼마나 빠르게 되는지 알수있다.
# bigger_model = keras.models.Sequential([
#         keras.layers.Dense(512,activation="relu", input_shape=(NUM_WORDS,)),
#         keras.layers.Dense(512,activation="relu"),
#         keras.layers.Dense(1,activation="sigmoid")
#         ])

# bigger_model.compile(optimizer="adam",
#                     loss="binary_crossentropy",
#                     metrics=["accuracy",'binary_crossentropy'])

# print(bigger_model.summary())


# bigger_history = bigger_model.fit(train_data, train_labels, epochs=20,
#                                  batch_size=512, validation_data=(test_data, test_labels), verbose=2)



#train_loss와 validation loss 그래프 그리기. 
#smaller_model이 bigger_model 보다 과대적합도 늦게 시작되고, 성능또한 천천히 감소한다.
def plot_history(histories, key="binary_crossentropy"):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],"--",label=name.title()+' Val')
        plt.plot(history.epoch , history.history[key], color=val[0].get_color(),label=name.title()+"Train")

        plt.xlabel('Epochs')
        plt.ylabel(key.replace('-',' ').title())
        plt.legend()

        plt.xlim([0, max(history.epoch)])
    plt.show()

# plot_history([('baseline', baseline_history), ('smaller',
#                                                smaller_history), ('bigger', bigger_history)])
# plot_history([('baseline', baseline_history), ('smaller',
#                                                smaller_history)])
#network의 용량이 많을수록 trainset을 더 빠르게 modeling할수 있는데 쉽게 overfitting된다.


##가중치를 규제하기

#오캄의 면도날 이론 : 어떤것을 설명하는 두방법이있으면 더 정확한 설명은 최소한의 가정이 필요한 간단한 설명,
#이와같이 간단한 모델은 복잡한 모델보다 과대적합이 더 작을 것이다.
#여기서 간단한 모델은 모델 params의 분포를 보았을때 엔트로피가 작은 모델. 그리고 적은 params를 가진 모델.
#따라서 가중치가 작은 값을 가지도록 network의 복잡도에 제약을 가하므로써 가중치 분포를 균일하게 만들어준다.
#network 손실함수에 큰 가중치에 해당하는 cost를 추가한다.

#L1 규제: 가중치의 절댓값에 비례하는 비용이 추가 (가중치의 L1 norm 이 추가된다.)
#L2 규제: 가중체의 제곱에 비례하는 비용이 추가된다(가중치의 L2 norm의 제곱을 추가)
#신경망에서는 L2 규제를 가중치 감쇠라고 부름.

#L1보다 L2규제를 더 많이쓰는데 이유는 일부 가중치 param을 L1은 0으로 만들기 때문

l2_model = keras.models.Sequential([
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dense(1, activation = "sigmoid")
        ])

l2_model.compile(optiizer="adam",loss='binary_crossentropy', metrics=['accuracy','binary_crossentropy'])

# l2_model_history = l2_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels),verbose= 2)
#위와같이 훈련을 시켜주면 l2(0.001)에 의해서 전체 손실에 층에있는 가중치행렬의 모든값이 0.001 * w **2 만큼 더해져서 패널티를 받는데 이로인해 train단계에서 test보다 손실이 더큼

# plot_history([('baseline', baseline_history),('l2',l2_model_history)])


#드롭아웃 추가하기
#dropout은 신경망에서 가장 효과적이고 널리사용하는 규제이다.
#keras에서는 dropout층을 이용해 드롭아웃을 해줄수있는데 이층은 바로 이전층의 출력에 dropout을 적용,

dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(NUM_WORDS, )),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

dpt_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[
                       'accuracy', 'binary_crossentropy'])

dpt_model_history = dpt_model.fit(
    train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

plot_history([('baseline', baseline_history), ('dropout', dpt_model_history)])