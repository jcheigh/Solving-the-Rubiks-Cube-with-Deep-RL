import tensorflow as tf
from tensorflow import keras 
from keras import layers

def make_model():
    input = keras.Input(shape= (480,))

    dense1 = layers.Dense(4096, activation ="elu")(input)
    dense2 = layers.Dense(2048, activation='elu')(dense1)

    value_layer = layers.Dense(512, activation="elu")(dense2)
    policy_layer = layers.Dense(512,activation="elu")(dense2)

    value = layers.Dense(1, name = "val")(value_layer)
    policy = layers.Dense(12, activation="softmax",
                          name = "policy")(policy_layer)

    model = keras.Model(inputs = input, outputs = [value, policy])
    return model

def compile_model(model):
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.001,
        decay_steps = 1000,
        decay_rate = 0.5
    )
    opt = keras.optimizers.RMSprop(learning_rate = lr_schedule)
    model.compile(loss = {'val' : 'mean_squared_error',
                          'policy': 'sparse_categorical_crossentropy'},
                  optimizer = opt)
    
    model.summary()