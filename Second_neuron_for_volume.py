from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Определение модели DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Дисконтный фактор
        self.epsilon = 1.0  # Параметр исследования
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(1, self.action_size+1)  # Случайное действие
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0]) + 1

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for index in minibatch:
            state, action, reward, next_state, done = self.memory[index]
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action-1] = target
            self.model.fit(state, target_f, epochs=20, verbose=2)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Параметры среды
state_size = 1  # Датасет временного ряда цен
action_size = 10  # Возможные значения объема от 1 до 10
n_prediction = 1  # Каждое предсказание делается через динамический шаг

# Инициализация DQN агента
agent = DQNAgent(state_size, action_size)

df = pd.read_excel('/content/train.xlsx')
df_t =  pd.read_excel('/content/test.xlsx')
train_data = df['Цена на арматуру']
test_data = df_t['Цена на арматуру'] # Тестовый датасет

# Создаем экземпляр MinMaxScaler
scaler = MinMaxScaler()

# Нормализуем данные из двух столбцов
train_data = scaler.fit_transform(df[['Цена на арматуру']])
test_data = scaler.fit_transform(df_t[['Цена на арматуру']])

batch_size = 2
# Обучение DQN агента
for i in range(len(train_data) - n_prediction):
    state = np.array([train_data[i]])
    action = agent.act(state)
    reward = -train_data[i+n_prediction-1] * action
    next_state = np.array([train_data[i+1]])
    done = False if i < len(train_data) - n_prediction - 1 else True
    agent.remember(state, action, reward, next_state, done)
    if done:
        break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Предсказание объема на тестовом датасете
predictions = []
for i in range(len(test_data) - n_prediction + 1):
    state = np.array([test_data[i]])
    action = agent.act(state)
    predictions.append(action)

print(predictions)  # Предсказанные объемы закупки на тестовом датасете
