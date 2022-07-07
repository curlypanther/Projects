#!/usr/bin/env python
# coding: utf-8

# # Градиентный бустинг

# ##### 1. Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше. Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('features.csv', index_col = 'match_id')
df.drop(['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']
           , axis=1, inplace=True)


# ##### 2. Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает число заполненных значений. Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски, и попробуйте для любых двух из них дать обоснование, почему их значения могут быть пропущены.

# In[2]:


a = []
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
    if round(pct_missing*100) > 0:
        a.append(col)


# In[3]:


a


# ### first_blood_player1 (игроки причастные к событию first_blood) - в 20% случаев первой крови нет в первые 5 минут игры
# ### first_blood_time (игровое время первой крови) - в 20% случаев первой крови нет в первые 5 минут игры

# ##### 3. Замените пропуски на нули с помощью функции fillna(). На самом деле этот способ является предпочтительным для логистической регрессии, поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание. Для деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение — в этом случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева. Также есть и другие подходы — например, замена пропуска на среднее значение признака. Мы не требуем этого в задании, но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой.

# In[4]:


df.fillna(0, inplace = True)


# ##### 4. Какой столбец содержит целевую переменную? Запишите его название.

# In[5]:


df['radiant_win']


# ##### Забудем, что в выборке есть категориальные признаки, и попробуем обучить градиентный бустинг над деревьями на имеющейся матрице "объекты-признаки". Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold), не забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени, и без перемешивания можно столкнуться с нежелательными эффектами при оценивании качества. Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации, попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30). Долго ли настраивались классификаторы? Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же качество, скорее всего, продолжит расти при дальнейшем его увеличении?

# In[9]:


import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plot

X_train = df.drop('radiant_win',axis = 1)
y_train = df.radiant_win
# Разделение выборки для кроссвалидации
cv = KFold(n_splits = 5,shuffle = True,random_state = 42)
# Кол-ва деревьев для модели
number_of_trees = [5,10,15,30,50,100,150]

# Список для записи результатов
scores = []

for tree in number_of_trees:
    model = GradientBoostingClassifier(n_estimators=tree, random_state=42)
    if tree == 30:
        start_time = datetime.datetime.now()
    model_scores = cross_val_score(model,X_train,y_train,cv = cv,scoring = 'roc_auc')
    if tree == 30:
        finish_time = datetime.datetime.now()
        print(f'time:{finish_time-start_time}')
    print(f'tree: {tree}, score: {np.mean(model_scores)}')
    scores.append(np.mean(model_scores))
plot.plot(number_of_trees, scores)
plot.xlabel('trees')
plot.ylabel('scores')
plot.show()    


# ### Кросс-валидация для градиентного бустинга с 30 деревьями заняла 0:02:42.645202. Показатель AUC-ROC равен 0.69

# ### Есть смысл использовать больше 30 деревьев, так как метрика увеличится. Для ускорения обучения можно уменьшить глубину деревьев (параметр max_depth)

# # Логистическая регрессия(2 задание)

# ##### 1. Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией) с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга. Подберите при этом лучший параметр регуляризации (C). Какое наилучшее качество у вас получилось? Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('features.csv', index_col = 'match_id')
df.drop(['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']
           , axis=1, inplace=True)
df.fillna(0, inplace = True)


# In[2]:


from sklearn.preprocessing import StandardScaler

cat_cols = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                      'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero','lobby_type']
columns_to_scale = df.columns.difference(['radiant_win', 'start_time'] + cat_cols)

data = df[columns_to_scale].values
scaler = StandardScaler()
data = scaler.fit_transform(data)
for i, column in enumerate(columns_to_scale):
    df[column] = data[:, i]


# In[3]:


columns_to_matrix = df.columns.difference(['radiant_win', 'start_time'])
x = df[columns_to_matrix].values
y = np.ravel(df['radiant_win'].values)


# In[4]:


x


# In[5]:


import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plot

cv = KFold(random_state = 42,shuffle = True,n_splits = 5)
C_range = [10.0 ** i for i in range(-5,4)]
scores = []
dic = {}
for C in C_range:
    print(f'C:{C}')
    start_time = datetime.datetime.now()
    model = LogisticRegression(random_state = 42, C = C)
    m_scores = cross_val_score(model,x,y,cv=cv,scoring = 'roc_auc')
    print(m_scores)
    print(np.mean(m_scores))
    scores.append(np.mean(m_scores))
    dic[C] = np.mean(m_scores)
    print('Time spent: ', datetime.datetime.now() - start_time)
plot.plot(range(-5,4), scores)
plot.xlim([-5,5])
plot.xlabel('log(C)')
plot.ylabel('scores')
plot.show()


# In[6]:


Maximum = max(scores)
Max_index = scores.index(Maximum)
print('C: ', C_range[Max_index], 'score: ', Maximum)


# ### Логистическая регрессия показывает высокое качество сравнимое с градиентным бустингом на числе дервьев более 150. При этом скорость обучения существенно выше.Как видно из графика, параметр регуляризации С в данном случае не оказывает существенного влияния на качество логистической регрессии. Наилучшее качество получается при C = 0.01: score = 0.71

# ##### Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является хорошей идеей. Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero, d1_hero, d2_hero, ..., d5_hero. Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации. Изменилось ли качество? Чем вы можете это объяснить?

# In[7]:


df_cat = df.copy(deep = True)
for column in cat_cols:
    df_cat.drop(column, axis = 1, inplace = True)


# In[8]:


columns_to_matrix = df_cat.columns.difference(['radiant_win', 'start_time'])
x = df_cat[columns_to_matrix].values
y = np.ravel(df_cat['radiant_win'].values)


# In[9]:


cv = KFold(random_state = 42,shuffle = True,n_splits = 5)
C_range = [10.0 ** i for i in range(-5,4)]
scores = []
dic = {}
for C in C_range:
    print(f'C:{C}')
    start_time = datetime.datetime.now()
    model = LogisticRegression(random_state = 42, C = C)
    m_scores = cross_val_score(model,x,y,cv=cv,scoring = 'roc_auc')
    print(m_scores)
    print(np.mean(m_scores))
    scores.append(np.mean(m_scores))
    dic[C] = np.mean(m_scores)
    print('Time spent: ', datetime.datetime.now() - start_time)
plot.plot(range(-5,4), scores)
plot.xlim([-5,5])
plot.xlabel('log(C)')
plot.ylabel('scores')
plot.show()


# In[10]:


Maximum = max(scores)
Max_index = scores.index(Maximum)
print('C: ', C_range[Max_index], 'score: ', Maximum)


# ### После удаления категориальных признаков качество практически не изменилось, значит они не имеют существенного значения для качества модели. Лучшее качество по-прежнему достигается при C = 0.01.

# ##### 3. На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают, какие именно герои играли за каждую команду. Это важные признаки — герои имеют разные характеристики, и некоторые из них выигрывают чаще, чем другие. Выясните из данных, сколько различных идентификаторов героев существует в данной игре (вам может пригодиться фукнция unique или value_counts).

# In[11]:


df = pd.read_csv('features.csv', index_col = 'match_id')


# In[12]:


len(df['r1_hero'].unique())


# ### Ответ: 108 героев 

# ##### 4. Воспользуемся подходом "мешок слов" для кодирования информации о героях. Пусть всего в игре имеет N различных героев. Сформируем N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче; единице, если i-й герой играл за команду Radiant; минус единице, если i-й герой играл за команду Dire. Ниже вы можете найти код, который выполняет данной преобразование. Добавьте полученные признаки к числовым, которые вы использовали во втором пункте данного этапа. 

# In[13]:


N = np.max(df['r1_hero'].unique())
x_pick = np.zeros((df.shape[0], N))

for i, match_id in enumerate(df.index):
    for p in range(5):
        x_pick[i, df.loc[match_id, 'r%d_hero' % (p+1)] - 1] = 1
        x_pick[i, df.loc[match_id, 'd%d_hero' % (p+1)] - 1] = -1


# In[14]:


for i in range(N):
    df_cat[f"Hero{i}"] = x_pick[:, i]


# In[15]:


df_cat


# ##### 5. Проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации. Какое получилось качество? Улучшилось ли оно? Чем вы можете это объяснить? 

# In[16]:


columns_to_matrix = df_cat.columns.difference(['radiant_win', 'start_time'])
x = df_cat[columns_to_matrix].values
y = np.ravel(df_cat['radiant_win'].values)


# In[17]:


cv = KFold(random_state = 42,shuffle = True,n_splits = 5)
C_range = [10.0 ** i for i in range(-5,4)]
scores = []
dic = {}
for C in C_range:
    print(f'C:{C}')
    start_time = datetime.datetime.now()
    model = LogisticRegression(random_state = 42, C = C)
    m_scores = cross_val_score(model,x,y,cv=cv,scoring = 'roc_auc')
    print(m_scores)
    print(np.mean(m_scores))
    scores.append(np.mean(m_scores))
    dic[C] = np.mean(m_scores)
    print('Time spent: ', datetime.datetime.now() - start_time)
plot.plot(range(-5,4), scores)
plot.xlim([-5,5])
plot.xlabel('log(C)')
plot.ylabel('scores')
plot.show()


# In[18]:


Maximum = max(scores)
Max_index = scores.index(Maximum)
print('C: ', C_range[Max_index], 'score: ', Maximum)


# ### Качество обучения существенно вoзросло. В данном случае лучший параметр C = 0.1. Это можно объяснить тем, что само значение идентификатора мало влияло на результат, значительно важнее был просто факт присутствия героя в игре

# ##### 6. Постройте предсказания вероятностей победы команды Radiant для тестовой выборки с помощью лучшей из изученных моделей (лучшей с точки зрения AUC-ROC на кросс-валидации). Убедитесь, что предсказанные вероятности адекватные — находятся на отрезке [0, 1], не совпадают между собой (т.е. что модель не получилась константной).

# ### Применим логистическую регрессию

# In[19]:


lr = LogisticRegression(C = 0.1)
lr.fit(x, y)


# In[20]:


df = pd.read_csv('features_test.csv', index_col = 'match_id')

df.fillna(0, inplace = True)

data = df[columns_to_scale].values
data = scaler.transform(data)
for i, column in enumerate(columns_to_scale):
    df[column] = data[:, i]
        
x_pick = np.zeros((df.shape[0], N))
for i, match_id in enumerate(df.index):
    for p in range(5):
        x_pick[i, df.loc[match_id, 'r%d_hero' % (p+1)] - 1] = 1
        x_pick[i, df.loc[match_id, 'd%d_hero' % (p+1)] - 1] = -1
for i in range(N):
    df[f"Hero{i}"] = x_pick[:, i]
    
for column in cat_cols:
    df.drop(column, axis = 1, inplace = True)


# In[21]:


df = df.drop('start_time',axis=1)


# In[22]:


proba = lr.predict_proba(df.values)


# In[23]:


proba[:,0].max()


# In[24]:


proba[:,0].min()


# In[25]:


proba[:,1].max()


# In[26]:


proba[:,1].min()


# In[27]:


len(proba[:,1])


# In[28]:


len(proba[:,0])


# ### Видно, что результаты адекватные 
