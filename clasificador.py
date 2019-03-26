import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv('cores.csv', index_col=0)
data = np.array(dataset[["Azul", "Verde", "Vermelho"]])

x_treino, x_val, y_treino, y_val = train_test_split(data, dataset['Classificacao'], test_size=0.30)

mSVM = SVC(kernel='linear')

mSVM.fit(x_treino, y_treino)

print("1 = Pele humana\n2 = Nao é um tom de pele humana\n")
c = 0
while c < len(x_val):
    print("Dado para classificação: {}".format(x_val[c]))
    resultado = mSVM.predict(x_val[c].reshape(1, -1))
    print('Resultado Predito: {}'.format(resultado))
    print('Resultado Correto: {}'.format(y_val.iloc[0]))
    c += 1
    print("\n")

y_pred_val = mSVM.predict(x_val)

print('Acurácia na validação: ')
print(accuracy_score(y_val, y_pred_val))

x_cores = np.array([[179, 206, 238], [113, 179, 60]])
y_cores = [1, 2]

print("Teste para a primeira cor: ", x_cores[0] , "\n")
resultado = mSVM.predict(x_cores[0].reshape(1, -1))
print("Resultado Predito: ", resultado)
print("Resultado Correto: ", y_cores[0], "\n")

print("Teste para a segunda cor: ", x_cores[1] , "\n")
resultado = mSVM.predict(x_cores[1].reshape(1, -1))
print("Resultado Predito: ", resultado)
print("Resultado Correto: ", y_cores[1], "\n")


print('Vetor de pesos: ', mSVM.coef_)

print('Bias: ', mSVM.intercept_)

print('Total de Vetores de Suporte em cada Classe: ', mSVM.n_support_, '\n')

print('Vetores de Suporte: \n')
print(mSVM.support_vectors_)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

for i in range(0,len(x_treino[:,0])):
    if (y_treino.iloc[i] == 1):
        ax.scatter(x_treino[i,0], x_treino[i,1], x_treino[i,2], s = 50, c='blue')
    else:
        ax.scatter(x_treino[i,0], x_treino[i,1], x_treino[i,2], s = 50, c='red')

coord_x = [10,150,300]
coord_y = [10,150,300]
coord_z = []

w1 = mSVM.coef_[0][0]
w2 = mSVM.coef_[0][1]
w3 = mSVM.coef_[0][2]
bias = mSVM.intercept_

for i in range(0,3):
    termo1 = (w1 * coord_x[i]) * -1
    termo2 = (w2 * coord_y[i]) * -1
    z = ((termo1 + termo2) - bias) / w3
    coord_z.append(int(z))

ax.plot(coord_x, coord_y, coord_z, c='grey', linewidth = 2)

ax.set_xlabel('Eixo X (B)')
ax.set_ylabel('Eixo Y (G)')
ax.set_zlabel('Eixo Z (R)')

plt.title('Classificação de Cores - Tons de Pele')

plt.show()
