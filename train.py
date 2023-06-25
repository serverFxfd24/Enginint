# файл с прогнозом. В процессе добавления считывания тестовой выборки
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from keras.optimizers import Adam,Adadelta,adadelta_legacy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_excel('/content/sample_data/train.xlsx')  # Замените 'data.csv' на путь к вашему файлу с данными
x = df['dt']
y = df['Цена на арматуру']
df_test = pd.read_excel('/content/sample_data/test.xlsx')
x_t = df_test['dt']
y_t = df_test['Цена на арматуру']
scaler = MinMaxScaler()
x = scaler.fit_transform(x.values.reshape(-1,1))
y = scaler.fit_transform(y.values.reshape(-1,1))
x_t = scaler.fit_transform(x_t.values.reshape(-1,1))
y_t = scaler.fit_transform(y_t.values.reshape(-1,1))
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(256, activation='relu', input_shape=(1,)),
  Dropout(0.2),#Dropout(0.19),Dropout(0.19),
  tf.keras.layers.Dense(256, activation='relu'),
  Dropout(0.2),
  tf.keras.layers.Dense(1)
])

model.summary()

learning_rate = 0.002  # desired learning rate value
optimizer = Adam(learning_rate=learning_rate)

# Компилируем модель
model.compile(loss='mse', optimizer=optimizer)

# Обучаем модель на нормализованных данных
model.fit(x, y, epochs=1000, batch_size=27)
from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_t)
# print(scaler.inverse_transform(y_predict))
# print(scaler.inverse_transform(y_t))
# Обратное преобразование нормализованных значений
#y_predict = scaler.inverse_transform(predictions)
mse = mean_squared_error(y_t, y_predict)
rmse = np.sqrt(mse)
# print(rmse)
import matplotlib.pyplot as plt
x = np.arange(len(y_predict))
plt.plot(x,y_t, color = 'g')
plt.plot(x,y_predict, color = 'r')
plt.show()
# print('y_t',y_t)
# print('y_predict',y_predict)
