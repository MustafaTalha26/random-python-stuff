from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def Sort(sub_li):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][1] > sub_li[j + 1][1]):
                tempo = sub_li[j]
                sub_li[j] = sub_li[j + 1]
                sub_li[j + 1] = tempo
    return sub_li

data = pd.read_csv('SongDetails.csv', index_col=0)
X = data.drop(['popularity'], axis=1)
y = data['popularity']

model = LinearRegression()

sc_x = StandardScaler()
Xscaled = sc_x.fit_transform(X)

model.fit(Xscaled, y)

importance = model.coef_

implist = []
for i,v in enumerate(importance):
    implist.append([X.columns[i],v])
implist = Sort(implist)
for feature in implist:
    print('Feature: %0s, Score: %.5f' % (feature[0],feature[1]))

print("Only genres: ")

data = pd.read_csv('SongDetails.csv', index_col=0)
genres = ['latin', 'rock', 'Dance/Electronic', 'metal','Folk/Acoustic', 
          'World/Traditional', 'classical', 'country', 'R&B', 'blues', 
          'pop', 'jazz', 'hip hop', 'set()', 'easy listening']
X = data[genres]
y = data['popularity']

model = LinearRegression()

sc_x = StandardScaler()
Xscaled = sc_x.fit_transform(X)

model.fit(Xscaled, y)

importance = model.coef_

implist = []
for i,v in enumerate(importance):
    implist.append([X.columns[i],v])
implist = Sort(implist)
for feature in implist:
    print('Genre: %0s, Score: %.5f' % (feature[0],feature[1]))

print("Without Genres: ")

data = pd.read_csv('SongDetails.csv', index_col=0)
genres = ['latin', 'rock', 'Dance/Electronic', 'metal','Folk/Acoustic', 
          'World/Traditional', 'classical', 'country', 'R&B', 'blues', 
          'pop', 'jazz', 'hip hop', 'set()', 'easy listening','popularity']
X = data.drop(genres, axis=1)
y = data['popularity']

model = LinearRegression()

sc_x = StandardScaler()
Xscaled = sc_x.fit_transform(X)

model.fit(Xscaled, y)

importance = model.coef_

implist = []
for i,v in enumerate(importance):
    implist.append([X.columns[i],v])
implist = Sort(implist)
for feature in implist:
    print('Score: %.5f, Genre: %0s' % (feature[1],feature[0]))