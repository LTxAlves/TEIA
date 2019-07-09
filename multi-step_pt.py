from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from matplotlib.lines import Line2D
from numpy import array


# converte dados seriais no tempo para um problema de aprendizado supervisionado
def serial_para_supervisionado(dados, n_in=1, n_out=1, tirar_Nan=True):
    n_vars = 1 if type(dados) is list else dados.shape[1]
    df = DataFrame(dados)
    cols, nomes = list(), list()
    # sequencia de entrada (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        nomes += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # sequence de previsao (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            nomes += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            nomes += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # juntar tudo
    agregado = concat(cols, axis=1)
    agregado.columns = nomes
    # tirar colunas com valor Nan
    if tirar_Nan:
        agregado.dropna(inplace=True)
    return agregado

# cria uma serie diferenciada
def diferenca(dataset, intervalo=1):
    diff = list()
    for i in range(intervalo, len(dataset)):
        value = dataset[i] - dataset[i - intervalo]
        diff.append(value)
    return Series(diff)

# transforma serie para treinamento e validação
def prepara_dados(serie, n_test, n_lag, n_seq):
    # extrair valores sem processamento
    valores_reais = serie.values
    # transforma dados para serem estacionários
    serie_diferente = diferenca(valores_reais, 1)
    valores_diferentes = serie_diferente.values
    valores_diferentes = valores_diferentes.reshape(len(valores_diferentes), 1)
    # rescala valores para intervalo ]-1, 1[
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(valores_diferentes)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transforma para supervisionado
    supervisionado = serial_para_supervisionado(scaled_values, n_lag, n_seq)
    valores_supervisionado = supervisionado.values
    # split into train and test sets
    treinamento, validacao = valores_supervisionado[0:-n_test], valores_supervisionado[-n_test:]
    return scaler, treinamento, validacao

# fit da LTSM para os dados de treinamento
def fit_lstm(treinamento, n_lag, n_seq, n_batch, num_epoca, n_neuronios):
    # muda o formato para [quant_dados, timesteps, features]
    X, y = treinamento[:, 0:n_lag], treinamento[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # cria a rede
    modelo = Sequential()
    modelo.add(LSTM(n_neuronios, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    modelo.add(Dense(y.shape[1]))
    modelo.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    while(num_epoca > 0):
        modelo.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        modelo.reset_states()
        num_epoca -= 1
    return modelo

# faz uma previsao com a LSTM
def prediz_lstm(modelo, X, n_batch):
    # muda o formato da entrada para [quant_dados, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # faz previsao
    previsao = modelo.predict(X, batch_size=n_batch)
    # converte para vetor
    return [x for x in previsao[0, :]]

# avalia o modelo da lstm
def avalia_modelo(modelo, n_batch, train, test, n_lag, n_seq):
    previsoes = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # prevê
        previsao = prediz_lstm(modelo, X, n_batch)
        # guarda previsão
        previsoes.append(previsao)
    return previsoes

# inverte previsao diferente
def diferenca_inversa(ultima_obs, previsao):
    # lista de previsao invertida
    invertida = list()
    invertida.append(previsao[0] + ultima_obs)
    # propaga diferença da previsao unsando primeiro valor da invertida
    for i in range(1, len(previsao)):
        invertida.append(previsao[i] + invertida[i-1])
    return invertida

# transforma inversamente daods das previsoes
def transformada_inversa(serie, previsoes, scaler, n_test):
    invertida = list()
    for i in range(len(previsoes)):
        # cria vetor a partir de previsoes
        previsao = array(previsoes[i])
        previsao = previsao.reshape(1, len(previsao))
        # inverte escala
        inv_scale = scaler.inverse_transform(previsao)
        inv_scale = inv_scale[0, :]
        # inverte diferenca
        index = len(serie) - n_test + i - 1
        ultima_obs = serie.values[index]
        inv_diferenca = diferenca_inversa(ultima_obs, inv_scale)
        # guarda
        invertida.append(inv_diferenca)
    return invertida

# avalia o erro quadrático médio pra cada timestep de previsao
def avalia_previsoes(validacao, previsoes, n_lag, n_seq):
    for i in range(n_seq):
        real = [row[i] for row in validacao]
        previsto = [forecast[i] for forecast in previsoes]
        rmse = sqrt(mean_squared_error(real, previsto))
        print('t+%d RMSE: %f' % ((i+1), rmse))

# plota as previsões sobre os dados originais
def plotar_previsoes(serie, previsoes, n_test):
    # plotar dados originais em azul
    pyplot.plot(serie.values, color='b')
    # plotar previsoes em vermelho
    for i in range(len(previsoes)):
        off_s = len(serie) - n_test + i - 1
        off_e = off_s + len(previsoes[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [serie.values[off_s]] + previsoes[i]
        pyplot.plot(xaxis, yaxis, color='r')
    linhas_legenda = [Line2D([0], [0], color='b', lw=1), Line2D([0], [0], color='r', lw=1)]
    pyplot.legend(linhas_legenda, ['Temperatura Real', 'Previsões'])
    # mostra o gráfico
    pyplot.show()

# carrega os dados
serie = read_csv("data/SP.csv", parse_dates = ['mdct'], index_col=0, usecols=['mdct', 'temp'], nrows=500)
# configura
n_lag = 1
n_seq = 3
n_test = 10
n_epochs = 50
n_batch = 1
n_neurons = 1
# prepare data
scaler, treinamento, validacao = prepara_dados(serie, n_test, n_lag, n_seq)
# fit do modelo
modelo = fit_lstm(treinamento, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# faz previses
previsoes = avalia_modelo(modelo, n_batch, treinamento, validacao, n_lag, n_seq)
# tranformada inversa de previsoes e validacao
previsoes = transformada_inversa(serie, previsoes, scaler, n_test+2)
real = [row[n_lag:] for row in validacao]
real = transformada_inversa(serie, real, scaler, n_test+2)
# avalia previsoes
avalia_previsoes(real, previsoes, n_lag, n_seq)
# mostra previsoes
plotar_previsoes(serie, previsoes, n_test+2)
