

### jupyter notebook 사용하기

`$ jupyter notebook`


### jupyter notebook에서 가상환경에 깔린 tf사용

`(venv)$ pip install ipykernel`

`$ python -m ipykernel install --user --name venv --display-name "myvenv"`


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
