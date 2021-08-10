import pickle
from sklearn.neural_network import MLPClassifier

mlp = pickle.load(open('mlp_model.sav', 'rb'))

#mlp.predict(...data...)