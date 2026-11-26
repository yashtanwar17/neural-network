import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

hours_studied = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10]).reshape(-1, 1)
exam_score = np.array([19, 24, 26, 33, 34, 41, 43, 50, 51, 58, 60, 66, 68, 75, 77, 84, 88, 95]).reshape(-1, 1)

time_train, time_test, score_train, score_test = train_test_split(hours_studied, exam_score, test_size=0.2)

model = LinearRegression()
model.fit(time_train,score_train)

print(model.score(time_test,score_test) *100)
