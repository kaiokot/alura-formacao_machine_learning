import csv

def carregar_acessos():
    X = []
    Y = []
    arquivo = open("acessos.csv", "rb")
    leitor = csv.reader(arquivo)

    leitor.next()

    for home, como_funciona, contato, comprou in leitor:
        dados = [int(home), int(como_funciona), int(contato)]
        X.append(dados)

        Y.append(int(comprou))
        
    return X, Y


def carregar_cursos():
    X = []
    Y = []
    arquivo = open("cursos.csv", "rb")
    leitor = csv.reader(arquivo)

    leitor.next()

    for home, como_funciona, contato, comprou in leitor:
        dados = [int(home), int(como_funciona), int(contato)]
        X.append(dados)

        Y.append(int(comprou))
        
    return X, Y
