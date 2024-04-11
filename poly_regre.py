import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# CSV 파일 로드
df = pd.read_csv('sydney_tem.csv', encoding='cp949')


# 데이터 확인
print(df.head())

# 특성과 타겟 설정
X = df['날짜'].values.reshape(-1, 1)  
y = df['9am 기온 (°C)'].values  


# 다항 특성 추가
degree = 3  # 다항식 차수
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# 다항 함수 출력
model = LinearRegression()
model.fit(X_poly, y)
print('Intercept:', model.intercept_) 
print('Coefficients:', model.coef_)

# 데이터 점 그리기
plt.scatter(X, y, color='blue', label='Data') 

# 회귀 선 그리기
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_pred = model.predict(X_plot_poly)
plt.plot(X_plot, y_pred, color='red', linewidth=2, label='Polynomial Regression')

plt.title('9am Wind Speed in Australia')
plt.xlabel('Day')
plt.ylabel('Wind Speed (km/h)')
plt.legend()
plt.show()
