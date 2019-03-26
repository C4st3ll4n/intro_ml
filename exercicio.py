import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv('iris.csv', index_col=0)

# print(dataset.isnull().any())

data = np.array(dataset[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])
# print(data)

xTreino, xValidacao, yTreino, yValidacao = train_test_split(data, dataset['Species'], train_size=0.30)

mSVM = SVC(kernel='linear')

mSVM.fit(xTreino, yTreino)

c = 0
while c < len(xValidacao):
    print("Dado para classificação: {}".format(xValidacao[c]))
    resultado = mSVM.predict(xValidacao[c].reshape(1, -1))
    print('Resultado Predito: {}'.format(resultado))
    print('Resultado Correto: {}'.format(yValidacao.iloc[0]))
    c += 1
    print("\n")

y_pred_val = mSVM.predict(xValidacao)

print('Acurácia na validação: {}'.format(accuracy_score(yValidacao, y_pred_val)))

print('Vetor de pesos: {}'.format(mSVM.coef_))

print('Bias: {}'.format(mSVM.intercept_))

print('Total de Vetores de Suporte em cada Classe: {}\n'.format(mSVM.n_support_))

print('Vetores de Suporte: {}'.format(mSVM.support_vectors_))

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

for i in range(0, len(xTreino[:, 0])):
    if yTreino.iloc[i] == 1:
        ax.scatter(xTreino[i, 0], xTreino[i, 1], xTreino[i, 2], s=50, c='black')
    else:
        ax.scatter(xTreino[i, 0], xTreino[i, 1], xTreino[i, 2], s=50, c='red')

coord_x = [10, 150, 300]
coord_y = [10, 150, 300]
coord_z = []

w1 = mSVM.coef_[0][0]
w2 = mSVM.coef_[0][1]
w3 = mSVM.coef_[0][2]
bias = mSVM.intercept_

for i in range(0, 3):
    termo1 = (w1 * coord_x[i]) * -1
    termo2 = (w2 * coord_y[i]) * -1
    z = ((termo1 + termo2) - bias) / w3
    coord_z.append(int(z))

ax.plot(coord_x, coord_y, coord_z, c='green', linewidth=2)

ax.set_xlabel('Eixo X ')
ax.set_ylabel('Eixo Y ')
ax.set_zlabel('Eixo Z ')

plt.title('Classificação de Iris')

plt.show()
