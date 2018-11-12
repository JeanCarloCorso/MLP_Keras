from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
import numpy as np

def PegaDados():
    dados = np.loadtxt("cancer.data", delimiter=",") # pega o dataset
    label_bruto = open("cancer-label.data", 'r')

    label = np.zeros(569).reshape((569))
    c = 0
    for l in label_bruto:
        #print(l)
        if(l == "M\n"):
            #print("entrou M")
            label[c] = 1
        elif(l == "B\n"):
            #print("entrou B")
            label[c] = 0
        c = c + 1
    label = label.astype(int)
    
    return dados, label

def normalization(X):
	#normalizacao
	for i in range(X.shape[1]):
		X[...,i] = (X[...,i] - np.min(X[...,i])) / (np.max(X[...,i])
		 - np.min(X[...,i]))
	
	return X

def particionar(arquivo, numeradaor, denominador, colunas):
    linhas, coluna_label = arquivo.shape
    coluna_label -= 1
    linhas = int(linhas/denominador * numeradaor)
    embaralhado = arquivo
    np.random.shuffle(embaralhado)

    treino = embaralhado[0:linhas,...]
    teste = embaralhado[linhas:,...]

    treino_dados = treino[...,0:colunas]
    treino_labels = treino[...,coluna_label:]    
    
    teste_dados = teste[...,0:colunas:]
    teste_labels = teste[...,coluna_label:]
	
    return treino_dados, treino_labels, teste_dados, teste_labels

# fix random seed for reproducibility
np.random.seed(7)

X, Y = PegaDados()
X = normalization(X)
integrada = np.concatenate((X,Y), axis=1) 
print("\n\n",integrada.shape)
treino_dados, treino_labels, teste_dados, teste_labels = particionar(integrada,2,3,31)

# create model
model = Sequential()
model.add(Dense(100, input_dim=30, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(33, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(11, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(treino_dados, treino_labels, epochs=200, batch_size=100)

# evaluate the model
scores = model.evaluate(teste_dados, teste_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
