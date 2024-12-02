# Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()

# Model
def define_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(8, activation='softmax'))
    return model

# Dataframe oluşturulması
df = pd.read_csv('train.csv')
tdf = pd.read_csv('test.csv')

dfbackup = label_encoder.fit_transform(df['NObeyesdad']) 

listco = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

df = pd.get_dummies(df.drop(['NObeyesdad','id'], axis=1),listco)
tdf = pd.get_dummies(tdf.drop(['id'], axis=1),listco)

df.insert(22, 'CALC_Always', False)
df['NObeyesdad'] = dfbackup

# Bağımlı ve bağımsı değişken ayrımı
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# MinMax ölçekleyici
sc_x = MinMaxScaler()
X = sc_x.fit_transform(X)
X_test = sc_x.transform(tdf)

# Veri seti eğitim ve test ayrımı
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12)

# Model oluşturulması, derlenmesi ve eğitilmesi
model = define_model()
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)
history = model.fit(
    X_train,
    y_train,
    epochs=60,
    batch_size=32, 
    validation_split=0.2, 
    verbose=2
)

# Test seti doğruluk hesaplaması
test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'Test Doğruluğu: {test_acc:.4f}')

# Sonuç gösterimi
history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1) 
melted_history = history_df.melt(id_vars='epoch', var_name='Metric', value_name='Value')
plt.figure(figsize=(6, 3), dpi=100)
sns.lineplot(x='epoch', y='Value', hue='Metric', data=melted_history)
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training & Validation Metrics')
plt.legend(title='Metrics')
plt.tight_layout()
plt.show()

# Test seti sonuçları
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# Karışıklık matrisi ve sınıflandırma raporu gösterimi
print(confusion_matrix(y_pred, y_val))
print(classification_report(y_pred, y_val))

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_pred = label_encoder.inverse_transform(y_pred)

indx = []
for x in range(len(y_pred)):
    indx.append([x+20758,y_pred[x]])

subdata = pd.DataFrame(indx, columns=['id','NObeyesdad'])

subdata.to_csv('submission.csv', sep=',', index=False, encoding='utf-8')




