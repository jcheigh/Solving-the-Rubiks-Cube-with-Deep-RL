import tensorflow as tf
from tensorflow import keras 
from keras import layers
import numpy as np

def make_model():
    """
    Creates keras.Model fully connected feed forward neural network. This value/policy network
    takes as input a tensor of shape (1,480) and outputs a value and a policy (array of len 12)

    Returns
    -------
    keras.Model
    Value/Policy Network
    """
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
    """
    Compiles model. Uses RMS prop and our loss functions are MSE for the value
    and categorical crossentropy for the policy

    Returns
    -------
    None
    Compiles Model
    """
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.001,
        decay_steps = 1000,
        decay_rate = 0.5
    )
    opt = keras.optimizers.RMSprop(learning_rate = lr_schedule)
    model.compile(loss = {'val' : 'mean_squared_error',
                          'policy': 'categorical_crossentropy'},
                  optimizer = opt)
    
    model.summary()
 
def test():
    model = make_model()
    compile_model(model)
    nn_input = np.random.randn(1,480)
    val, policy = model.predict(nn_input)
    print(f"Value: {val}")
    print(f"Policy: {policy}")

    assert val.shape == (1,1)
    assert policy.shape == (1,12)
    assert np.round(np.sum(policy[0])) == 1

if __name__ == "__main__":
    test()