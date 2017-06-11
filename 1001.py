import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv1D
from keras.layers.normalization import BatchNormalization
from keras import optimizers

train_set = {
    7111: 0, 8809: 6, 2172: 0, 6666: 4, 1111: 0,
    2222: 0, 7662: 2, 9313: 1, 0000: 4, 5555: 0,
    8193: 3, 8096: 5, 4398: 3, 9475: 1, 9038: 4,
    3148: 2
}
test_set = {
    2889: 5
}
os = {
    0: 1, 1: 0, 2: 0, 3: 0, 4: 0,
    5: 0, 6: 1, 7: 0, 8: 2, 9: 1
}


def create_train_set(n):
    my_train_set = {}
    for i in range(n):
        while True:
            r = np.random.randint(0, 9999)
            if r not in test_set.keys():
                break
        y = np.sum([os[int(a)] for a in list(str(r).zfill(4))])
        my_train_set[r] = y
    return my_train_set


def nd(t_set):
    return [np.expand_dims(
                np.array([list(str(k).zfill(4)) for k in t_set.keys()]).astype(int),
                2),
            np.array([[k] for k in t_set.values()]).astype(int)]


train_set.update(create_train_set(50))
x_train, y_train = nd(train_set)
x_test, y_test = nd(test_set)

model = Sequential()
model.add(Conv1D(16, 1, input_shape=[4, 1]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(16, 1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(8, 1))
model.add(Activation('relu'))
model.add(Conv1D(8, 1))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(units=1))

adam = optimizers.Adam(lr=1e-3)

def acc(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))

model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=[acc])

model.fit(x_train, y_train, epochs=3000, batch_size=1000)

y_train_pred = model.predict(x_train, batch_size=1000)
y_test_pred = model.predict(x_test, batch_size=1000)

print(test_set)
print(np.round(y_test_pred))

