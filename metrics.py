from tensorflow.keras import backend as K

def f1(precision, recall):
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

