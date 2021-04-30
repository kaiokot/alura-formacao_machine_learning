from sklearn.naive_bayes import MultinomialNB
from dados import carregar_acessos

X, Y = carregar_acessos()

treino_dados = X[:90]
treino_marcacoes = Y[:90]

teste_dados = X[-9:]
teste_marcacoes = Y[-9:]


modelo = MultinomialNB()

modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)

total_testes = len(teste_dados)

acuracia = 100.0 * total_de_acertos / total_testes


print(acuracia)
