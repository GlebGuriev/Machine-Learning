import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.width = None

# читаем таблицы из файлов
events_data = pd.read_csv('datasets/event_data_train.csv')
submissions_data = pd.read_csv('datasets/submissions_data_train.csv')

# добавим колонки с датой и днями
events_data['date'] = pd.to_datetime(events_data['timestamp'], unit='s')
events_data['day'] = events_data['date'].dt.date
submissions_data['date'] = pd.to_datetime(submissions_data['timestamp'], unit='s')
submissions_data['day'] = submissions_data['date'].dt.date

# создадим датафрейм, в котором посчитаем количество действий пользователя
users_events_data = \
    events_data.pivot_table(index='user_id',
                            columns='action',
                            values='step_id',
                            aggfunc='count',
                            fill_value=0).reset_index()

# датафрейм, в котором посчитаем количество правильных и неправильных сабмишнов
users_score = submissions_data.pivot_table(index='user_id',
                                           columns='submission_status',
                                           values='step_id',
                                           aggfunc='count',
                                           fill_value=0)

# количество уникальных дней, в которые пользователь был активен
users_days = events_data.groupby('user_id')['day'].nunique().to_frame().reset_index()

# максимальное количество дней между посещениями ресурса
gap_data = events_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day']) \
                                                       .groupby('user_id')['timestamp'] \
                                                       .apply(list) \
                                                       .apply(np.diff).values
gap_data = pd.Series(np.concatenate(gap_data, axis=0))
gap_data = gap_data / (24 * 60 * 60)
# print(gap_data.quantile(0.95))

now = 1526772811
drop_out_threshold = 2592000

# users_data - датафрейм со всеми необходимыми данными
users_data = events_data.groupby('user_id', as_index=False) \
                        .agg({'timestamp': 'max'}) \
                        .rename(columns={'timestamp': 'last_timestamp'})
users_data['is_gone_user'] = (now - users_data['last_timestamp']) > drop_out_threshold
users_data = users_data.merge(users_score, on='user_id', how='outer').fillna(0)
users_data = users_data.merge(users_events_data, on='user_id', how='outer').fillna(0)
users_data = users_data.merge(users_days, on='user_id', how='outer').fillna(0)
users_data['passed_course'] = users_data['passed'] > 170

'''Part II - Module 2.10
Готовим данные для машинного обучения, создаем тренировочные сеты'''

print(users_data[users_data['passed_course']].day.median())

user_min_time = (events_data.groupby('user_id', as_index=False).
                 agg({'timestamp': 'min'}).
                 rename({'timestamp': 'min_timestamp'}, axis=1))
users_data = users_data.merge(user_min_time, how='outer')

events_data_train = events_data.merge(users_data[['user_id', 'min_timestamp']], on='user_id', how='left') \
    .query("(timestamp - min_timestamp) <= (3 * 24 * 60 * 60)")

submissions_data_train = submissions_data.merge(users_data[['user_id', 'min_timestamp']], on='user_id', how='left') \
    .query("(timestamp - min_timestamp) <= (3 * 24 * 60 * 60)")


''' Задача: bad_step_data = submissions_data.pivot_table(index='step_id',
                                             columns='submission_status',
                                             values='user_id',
                                             aggfunc='count')
max_wrongs = bad_step_data['wrong'].max()
print(bad_step_data.query('wrong == @max_wrongs'))'''

"""Начало машинного обучения"""

X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index(). \
            rename(columns={'day': 'days'})
X = X.merge(submissions_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index().
            rename(columns={'step_id': 'step_tried'}), on='user_id', how='outer')
X = X.merge(submissions_data_train.pivot_table(index='user_id',
                                               columns='submission_status',
                                               values='step_id',
                                               aggfunc='count',
                                               fill_value=0), on='user_id', how='outer')
X['correct_ratio'] = X.correct / (X.correct + X.wrong)
X = X.merge(events_data_train.pivot_table(index='user_id',
                                          columns='action',
                                          values='step_id',
                                          aggfunc='count',
                                          fill_value=0).reset_index()[['user_id', 'viewed']], how='outer')
X = X.fillna(0)
X = X.merge(users_data[['user_id', 'passed_course', 'is_gone_user']], how='outer')
X = X[~((X.is_gone_user == False) & (X.passed_course == False))]
y = X.passed_course
X = X.drop(['passed_course', 'is_gone_user'], axis=1)
X = X.set_index(X.user_id)
X = X.drop('user_id', axis=1)
"""Экспорт подготовленных данных в csv"""
X.to_csv('datasets/X_data.csv')
y.to_csv('datasets/Y_data.csv')
