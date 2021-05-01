from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from dados import carregar_buscas

df = pd.read_csv("buscas.csv")

X_df = df[["home", "busca", "logado"]]
Y_df = df["comprou"]


Xdummies = pd.get_dummies(X_df)
Ydummies = Y_df

X = Xdummies.values
Y = Ydummies.values


tamanho_de_treino = int(0.9 * len(Y))

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

tamanho_de_teste = len(Y) - tamanho_de_treino
teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]


def fit_and_predict(algoritmo, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    diferencas = resultado - teste_marcacoes

    acertos = [d for d in diferencas if d == 0]
    total_de_acertos = len(acertos)

    total_testes = len(teste_dados)
    acuracia = 100.0 * total_de_acertos / total_testes

    print("Taxa de acerto algoritmo {0} : {1}".format(algoritmo, acuracia))


modelo = MultinomialNB()
fit_and_predict("MultinomialNB", modelo, treino_dados, treino_marcacoes,
                teste_dados, teste_marcacoes)


modelo = AdaBoostClassifier()
fit_and_predict("AdaBoostClassifier", modelo, treino_dados, treino_marcacoes,
                teste_dados, teste_marcacoes)


acerto_de_um = sum(treino_marcacoes)
acerto_de_zeros = len(treino_marcacoes) - acerto_de_um
taxa_de_acerto_base = 100.0 * \
    max(acerto_de_um, acerto_de_zeros) / len(treino_marcacoes)

print("Taxa de acerto baseline %2f" % taxa_de_acerto_base)
