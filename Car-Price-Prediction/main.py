from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import regularizers

data = pd.read_csv('cardekho.csv', index_col=0)
print(data.isnull().sum())
data = data.dropna()
one_hot_encoded_data = pd.get_dummies(data, columns = ['fuel', 'seller_type','transmission','owner'], dtype='int')
print(one_hot_encoded_data.isnull().sum())

X = one_hot_encoded_data.drop(['selling_price'], axis=1)
y = one_hot_encoded_data['selling_price']

inputdim = len(X.columns)

sc_x = StandardScaler()
Xscaled = sc_x.fit_transform(X)

def define_model(input_size):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2048,input_dim=input_size, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1028, activation='relu',
                                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                bias_regularizer=regularizers.L2(1e-4),
                                activity_regularizer=regularizers.L2(1e-5)))
    model.add(keras.layers.Dense(1028, activation='relu',
                                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                bias_regularizer=regularizers.L2(1e-4),
                                activity_regularizer=regularizers.L2(1e-5)))
    model.add(keras.layers.Dense(1028, activation='relu',
                                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                bias_regularizer=regularizers.L2(1e-4),
                                activity_regularizer=regularizers.L2(1e-5)))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    return model

X_train, X_test, y_train, y_test = train_test_split(Xscaled, y, test_size=0.33, random_state=42)

model = define_model(inputdim)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='mae'
)
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128, 
    validation_split=0.2, 
    verbose=2
)

y_pred = model.predict(X_test)

dftest = pd.DataFrame(y_test)
dftest.insert(1,'prediction',y_pred,True)
dftest.to_csv('out.csv', index=True)

plt.plot(X_test[:,0], y_test, 'o')
plt.plot(X_test[:,0], y_pred, '.')
plt.show()

