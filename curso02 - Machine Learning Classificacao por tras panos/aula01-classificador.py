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
animal_misterioso4 = [1, 0, 1]

previsoes = [animal_misterioso1, animal_misterioso2,
             animal_misterioso3, animal_misterioso4]


testes = [1, -1, 1, 1]
             

previsao = modelo.predict(previsoes)

print(previsao)
