

### jupyter notebook 사용하기

`$ jupyter notebook`


### jupyter notebook에서 가상환경에 깔린 tf사용

`(venv)$ pip install ipykernel`

`$ python -m ipykernel install --user --name venv --display-name "myvenv"`

### verbose mode

0 = silent, 
1 = progress bar, 
2 = one line per epoch.

### Segmentation fault
저장하는 file 과 불러오는 file의 경로가 다를 경우 발생했었음.
# quickstart

기본적인 mnist를 이용한 예제인데 begin에서는 주어진 keras의 층을 하나하나 쌓아나가고
advance는 모델을 만드는데 조금더 custom해서 만드는 과정을 보는 예제이다.


# fashion_mnist

옷의 종류를 구분하는 mnist와 같은 분류 tutorial 이다. 해당 tutorial을 통해서
간단하게 train_data 와 test_data의 구조를 살펴보고
argmax 와 softmax를 한번더 써보는 tutorial이었다.
뿐만아니라 plt 를 이용해 실제로 data를 예측하고 시각적으로 확인할수있는
mnist2 tutorail이다.

# moview_review

### movie_review_classify을 통해
tf_hub에서 embedding layer를 가져와서 영화리뷰에 대한 긍정 부정을 평가하는 모델을 만드는 실습이었는데
```python
(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(('train[:60%]', 'train[60%:]'),tfds.Split.TEST),
    as_supervised=True,
)
```
이 부분에서 tf2 공홈에서 보여주는 split API부분이 deprecated 되어서 새로운 split API를 적용한다고 애를 먹었다.



### pretrained movie_dataset tutorial을 통해 
embedding층을 사용하기 위해 입력된 data의 길이를 균일하게 만드는 padding에 대해서 사용해보았고
헬퍼함수 작성을 통해 word가 정수로 전처리 되어있는 부분을 다시금 문장으로 변환하는 처리도 해보았다.
특히나 마지막에 matplot을 이용한 학습과정의 시각화를 통해서 graph를 확인할수 있었고, terminal이나 그래프에서 overfitting을 확인하는 튜토리얼이었다.


# 자동차연비 예측하기 (regression)

회귀(regression)은 가격이나 확률처럼 연속된 값을 예측하는데 사용된다.

해당 튜토리얼은 pandas로 데이터를 읽어오고 df로 변환해서 사용하기때문에 기본적인 pandas사용을 경험해볼 수 있고, seaborn을 통해서 시각화를 해볼 수 있다.
뿐만아니라 normalization과 Early stopping을 사용해볼 수 있는 튜토리얼이었다.



# overfitting & underfitting 

해당 튜토리얼에서는 overfitting과 underfitting에 대해서 간략하게 설명해주고
overfitting을 막는 방법에 대해서 설명해준다.

* 보다 많은 훈련 data를 사용한다.
* 네트워크의 용량을 줄인다(dense가 받는 param의 양을 줄인다)
* 가중치 규제를 추가한다.(L1,L2// 그런데 L2규제를 많이쓴다.)
* Dropout을 추가한다(가장 효과적이고 많이 쓰는 방법인데 Dropout을 한다는 것은 특정 node를 학습시키지 않는다는 뜻이고 이 비율은 보통 (0.2~0.5가 된다)(테스트시에는 모든 유닛들이 활성화 되기때문에 균형을 맞추기위해 layer의 출력값을 dropout의 비율만큼 줄인다.)

> 해당 예제를 통해서 가중치를 규제하는 경우는 아직 이해가 어려운 부분이 많지만 모델의 용량을 조절하는 것이나, dropout을 이용해서
> overfitting의 타이밍을 늦추거나 이후에 일어나는 성능저하는 천천히 가져갈수 있는 연습을 해보았다.



# save and recovery

드디어 기다리고 기다리던 학습한 or 학습중인 model의 저장과 불러오기 기능을 배웠다.
물론 tf.keras에서 지원하는 기능을 활용한 부분이지만
check_point를 정해서 period 마다 저장하기, 가중치만 저장하기, model전체를 저장하는 방법을 알아보고 실습해볼수 있었다.