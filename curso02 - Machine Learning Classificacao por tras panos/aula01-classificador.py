from sklearn.naive_bayes import MultinomialNB

# eh gordo? perna curta? faz au au?
# 1 = Cachorro , -1 = Porco

cachorro1 = [1, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [0, 1, 1]
porco1 = [1, 1, 0]
porco2 = [1, 0, 0]
porco3 = [0, 1, 0]

dados = [cachorro1, cachorro2, cachorro3, porco1, porco2, porco3]
teste = [1, 1, 1, -1, -1, -1]

modelo = MultinomialNB()
modelo.fit(dados, teste)

animal_misterioso1 = [1, 1, 1]
animal_misterioso2 = [0, 1, 0]
animal_misterioso3 = [0, 1, 1]


testes = [animal_misterioso1, animal_misterioso2,
          animal_misterioso3]


marcacoes_teste = [1, -1, -1]

resultado = modelo.predict(testes)
print(resultado)

diferencas = resultado - marcacoes_teste
print(diferencas)

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
print(total_de_acertos)

total_testes = len(testes)

acuracia = 100.0 * total_de_acertos / total_testes

print(acuracia)
