# 현재 대학교 수와 수능 응시생 수-> 수능 응시생 수에 따라 바뀌되는 대학교 수 확인
# 지난 20년간 수험생 수와 대학 수 추이 확인
# 이를 통해 감소한 수험생 수에 따른 대학 수 감소의 관계성 확인
# 이 자료들로 20년 뒤 수험생 수를 예측하고, 감소한 수험생 수에 따른 대학 수 예측


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 나열
# 2024년부터 2006년까지 2년 간격으로 대학수학능력시험 지원자 수
# 2024년부터 2006년까지 2년 간격으로 일반대+전문대+교육대+산업대를 모두 포함한 대학 수

# years 범위 설정
years = np.array([2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]).reshape(-1, 1)
candidates = np.array([593806, 584934, 677834, 693631, 650747, 631187, 593527, 548734, 509821, 504588])
universities = np.array([352, 344, 345, 343, 340, 339, 340, 339, 336, 332])


# 지원자 수의 20년동안의 변화 추이를 Linear Regression model로 확인
candidates_model = LinearRegression()
candidates_model.fit(years, candidates)


# 20년 동안의 대학 수 변화 확인
universities_model = LinearRegression()
universities_model.fit(years, universities)


# 대학교 수와 수험생 수의 관계성을 알아보기 위한 코드 요청
# 20년 동안 감소한 수험생의 수를 x축의 독립 변수로, 이에 따른 감소한 대학 수는 종속 변수로 설정
# Linear Regression model로 수험생 수와  사이의 관계성을 구함
# ->Linear Regression model을 이용한 코드 제시
candidates_change = candidates[-1] - candidates[0] # 2006년에서 2024년까지의 수험생 수 변화
universities_change = universities[-1] - universities[0] # 2006년부터 2024년까지의 대학 수 변화

# Linear Regression model을 이용하여 관계성 학습(Chat gpt 제시)
relationship_model = LinearRegression()

# 독립 변수는 수험생 수로, 이에 따른 종속변수는 대학 수로 설정한다.
relationship_model.fit(candidates.reshape(-1, 1), universities)

print(f"수험생 수와 대학 수 감소에 따른 기울기: {relationship_model.coef_[0]}")
print(f"절편 (Intercept): {relationship_model.intercept_}")


# 20년 뒤의 수험생 수를 위에서 학습한 모델로 예측한다.
future_2044 = 2044
predicted_candidates_future = candidates_model.predict(np.array([[future_2044]]))

# 20년 뒤인 2044를 future year로 설정하여 predicted
print(f"2044년의 예상 수험생 수 : {predicted_candidates_future[0]}")

# 2044년에 예상되는 수험생 수와 위에서 구한 수험생과 대학 수 사이의 관계성을 이용해
# 2044년에 남은 대학 수를 예측한다.
predicted_universities_future = relationship_model.predict(np.array([[predicted_candidates_future[0]]]))

print(f"2044년의 예상 대학 수 : {predicted_universities_future[0]}")

# 학습한 모델에 대한 성능 확인을 위한 코드 요청
# -> MSE 방식의 코드 제공(Chat gpt)
# MSE를 계산하기 위해 reshape로 차원과 형상을 맞춘다.
universities_pred = relationship_model.predict(candidates.reshape(-1, 1))
mse = mean_squared_error(universities, universities_pred)
print(f"Mean Squared Error: {mse}")

# 결과를 그래프로 시각화하기

# x축을 시간 축으로, y축에 두 가지 데이터를 넣기 위해 그래프 세분화
# 좀 더 명확한 구분을 위한 방법 요청 -> zorder를 이용하여 우선시되는 데이터의 가시성 조정(chat gpt)
fig, ax1 = plt.subplots(figsize=(10,6))
# 한 그래프 내에 두 개의 y축을 나타내기 위해 코드 요청-> 축을 나눈 형태 제시(chat gpt)

# 첫 번째 y축은 대학 축에 관하여 설정
ax1.scatter(years, universities, color='blue', label="Number of universities", zorder=5)
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of universities", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 두 번째 y축은 수험생 수에 관하여 설정
ax2 = ax1.twinx() # 두 번째 y축 생성
ax2.scatter(years, candidates, color='navy', label="Number of candidates", zorder=5)
ax2.set_ylabel("Number of candidates", color='navy')
ax2.tick_params(axis='y', labelcolor='navy')

# 예측된 2044년의 대학 수와 수험생 수를 나타내기
ax1.scatter(2044, predicted_universities_future, color='yellow', label='Predicted universities in 2044', zorder=10)
plt.scatter(2044, predicted_candidates_future, color='orange', label='Predicted candidates in 2044', zorder=10)

# 레전드 설정
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Number of universities and candidates over the years")
ax1.grid(True)
plt.show()

# 실제 데이터를 기반으로 한 수험생 수와 대학 수의 관계성을 그래프로 나타내기
# Linear Regression model로 학습한 관계성 나타내기
plt.figure(figsize=(10, 6))
relationship_values = relationship_model.predict(candidates.reshape(-1,1))
plt.plot(candidates, relationship_values, color='green', label='Relationship between candidates and universities')
plt.xlabel("Number of candidates")
plt.ylabel("Number of universities")
plt.title("Relationship between the number of candidates and universities")
plt.legend()
plt.grid(True)
plt.show()

# 학점 : A0
# 스스로 완전한 코드를 짜기에는 부족하더라도 그동안 배운 내용들을 활용해 원하는 방향으로 코딩을 할 수 있게 되었습니다.
# Hello world도 잘 모르는 상태로 시작했으나 chat gpt를 이용하여 필요한 형태의 코딩 결과를 얻을 수 있게 되었습니다.
# 주제를 정하고 여러 모델을 시도하는 과정에서 대략적인 모델들의 특징과 용도의 차이를 이해할 수 있습니다.
# 코드와 오류가 어떤 내용인지 이해하고 간단히 수정할 수 있게 되었습니다.
# 따라서 A0 학점을 부여하고 싶습니다.