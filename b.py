import pandas as pd
from ml import *
from graphs import *


def predicao():
    return



def avaliacaoSimilaridade(modelo, dados, janela):

    """
    Descrição:

    Entrada(s):

    Saída(s):
    """


    def treinamento(modelo, vetor, linhaLatente, col):
        modelo.fit(vetor, vetor, epochs=100, batch_size=128, verbose=0)
        loss, mae, mse = modelo.evaluate(vetor, vetor)
        latent = modelo.encoded(vetor)
        linhaLatente.append(latent.numpy().tolist()[0])
        #print(f'{col}: {loss}, {vetor}, {latent}')
        return loss, latent


    def predicao(modelo, vetor, linhaLatente, linhaPerdas, perda, col):
        loss, latent = treinamento(modelo, vetor, linhaLatente, col)
        perdaRelativa = np.abs(loss - perda)/perda
        linhaPerdas.append(perdaRelativa)
        return perdaRelativa, latent


    print(f'Dados:\n{dados}\n')
    espacoLatente, perdas = list(), list()
    for index, col in enumerate(dados.columns[:10], 0):
        linhaLatente, linhaPerdas = list(), list()
        vetor = np.array(dados.loc[:, col]).reshape(1, janela)
        perda, latent = treinamento(modelo, vetor, linhaLatente, col)
        linhaPerdas.append(0)
        for index2, col2 in enumerate(dados.columns[index+1:], index+1):
            vetor2 = np.array(dados.loc[:, col2]).reshape(1, janela)
            predicao(modelo, vetor2, linhaLatente, linhaPerdas, perda, col2)
        print('\n')
        espacoLatente.append(linhaLatente)
        perdas.append(linhaPerdas)
    return espacoLatente, perdas


nsdq = pd.read_csv(r'finance\Retornos\NSDQ.txt', sep=',')
sp500 = pd.read_csv(r'finance\Retornos\S&P500.txt', sep=',')
dataframe = pd.concat([nsdq, sp500], axis=1)
dataframe = dataframe.loc[:19, dataframe.columns[:25]]
normalizaDados(dataframe, False)

janela = 20
model = NeuralNetwork(janela, 3)
model.compile(optimizer='adam', loss=NeuralNetwork.customLoss, metrics=['mae', 'mse'])
espacoLatente, perdas = avaliacaoSimilaridade(model, dataframe.loc[:janela-1, dataframe.columns], janela)
#print(espacoLatente, perdas)


for index, ticker in enumerate(dataframe.columns[:10], 0):
    scatter(dataframe.columns, ticker, *np.transpose(espacoLatente[index]))
