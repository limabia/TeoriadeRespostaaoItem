# -*- coding: utf-8 -*-
# Bianca L. Santos, Cleiton Dantas, Fábio Oliveira , Mariana Santana
import math

import numpy
import pandas
from scipy import optimize
from scipy import stats


path = '../dados/'

""" numero de simulações a serem feitas"""
n_simulacoes = 1000

"""importa as questoes do arquivo passado pelo prof """
tabela_questoes = pandas.read_table(path+'questoes.txt', sep=' ', names=('a', 'b'))
respostas = pandas.read_table(path+'respostas.txt', sep=' ', header=None)
alunos = [-1.0, -0.5, 0.0, 0.5, 1.0]


def calcula_pr_questoes():
    for i in range(0, 5):
        tabela_questoes['pr_' + str(i)] = calcula_pr(alunos[i], tabela_questoes)


def f(a, b, theta):
    return math.exp(a*(theta - b)) / (1 + math.exp(a*(theta - b)))


def f_prime(a, b, theta):
    """ derivada da funcao """
    return (a * math.exp(a*(theta - b))) / ((1 + math.exp(a*(theta - b))) ** 2)


def f_prime2(a, b, theta):
    """ derivada da funcao """
    return (a * a * math.exp(a*(theta - b))) / ((1 + math.exp(a*(theta - b))) ** 4)


def log_likelihood_prime(theta, questoes, respostas):
    """Calcula a derivada do log likelihood"""
    soma = 0
    for i in range(0, len(questoes)):
        x = respostas.iloc[i]
        a = questoes.iloc[i, 0]
        b = questoes.iloc[i, 1]
        if x == 1:
            soma += f_prime(a, b, theta) / f(a, b, theta)
        else:
            soma -= f_prime(a, b, theta) / (1 - f(a, b, theta))
    return soma


def log_likelihood_prime2(theta, questoes, respostas):
    """Calcula a derivada do log likelihood"""
    soma = 0
    for i in range(0, len(questoes)):
        x = respostas.iloc[i]
        a = questoes.iloc[i, 0]
        b = questoes.iloc[i, 1]
        if x == 1:
            soma += (f_prime2(a, b, theta) * f(a, b, theta) - ((f_prime(a, b, theta) ** 2))) / (f(a, b, theta) ** 2)
        else:
            soma -= (f_prime2(a, b, theta) * (1 - f(a, b, theta)) + ((f_prime(a, b, theta) ** 2))) / ((1 - f(a, b, theta)) ** 2)
    return soma


def saida_3_1():
    """Encontra a habilidade para cada um dos 2000 alunos e salve em um arquivo dados/II1.txt """
    print("inicio do item 3.1")
    saida31 = open(path+'II1.txt', 'w')
    for i in range(2000):
        # estima o theta usando metodo da secante
        theta_chapeu = calcula_theta_chapeu(tabela_questoes, respostas.iloc[:, i])
        print('theta estimado %i: %f' % (i, theta_chapeu))
        numpy.savetxt(saida31, [theta_chapeu], fmt='%.5f')
    saida31.close()
    print("fim item 3.1")


def calcula_theta_chapeu(questoes, respostas_aluno):
    try:
        theta_chapeu = optimize.newton(log_likelihood_prime, 0.0, args=(questoes, respostas_aluno), fprime=log_likelihood_prime2)
    except OverflowError:
        theta_chapeu = float("inf")
    except RuntimeError:
        theta_chapeu = -float("inf")
    return theta_chapeu


def compara_aluno5(notas_prova, vezes_aluno5_melhor):
    """Compara a nota do aluno 5 com os demais alunos em casa uma das provas e retorna """
    nota_aluno5 = notas_prova[4]
    # soma a quantidade de vezes que o aluno 5 foi melhor do que os demais
    for i in range(0, 4):
        if nota_aluno5 > notas_prova[i]:
            vezes_aluno5_melhor[i] += 1
    return vezes_aluno5_melhor


def simula_provas(tamanho_prova, quantidade_simulacoes, gerador_provas):
    """ Efetivamente entrar no problema:
    Chama as funções necessárias para dar a probabilidade do aluno 5 ser melhor que os demais """
    # habilidades de cada um dos 5 alunos
    vezes_aluno5_melhor = numpy.zeros(4)

    # para um numero alto de vezes, verifica quantas vezes o aluno 5 foi melhor em relação aos demais
    for i in range(quantidade_simulacoes):
        prova_aleatoria = gerador_provas(tamanho_prova)
        notas_prova = calcula_notas_alunos(prova_aleatoria)
        compara_aluno5(notas_prova, vezes_aluno5_melhor)
        print(i)
    probabilidade_aluno5_melhor = vezes_aluno5_melhor / quantidade_simulacoes
    return probabilidade_aluno5_melhor


def calcula_notas_alunos(prova):
    """ calcula o desempenho dos alunos para a prova passada """
    notas_prova = numpy.zeros(5)
    # para cada um dos 5 alunos
    for i in range(0, 5):
        respostas_aluno = []
        # calcula a probabilidade de acerto de todas as questoes da prova para o aluno i
        for p in prova['pr_' + str(i)]:
            respostas_aluno.append(numpy.random.binomial(1, p))
        notas_prova[i] = calcula_theta_chapeu(prova, pandas.Series(respostas_aluno))
    # retorna a nota de cada aluno
    # print(notas_prova)
    return notas_prova


def calcula_notas_alunos2(prova):
    """ calcula o desempenho dos alunos para a prova passada """
    notas_prova = numpy.zeros(5)
    # para cada um dos 5 alunos
    for i in range(0, 5):
        # calcula a probabilidade de acerto de todas as questoes da prova para o aluno i
        for p in prova['pr_' + str(i)]:
            # aplica a distribuicao binomial em cima da probabilidade de acerto para o aluno i e assim calcula sua nota
            notas_prova[i] += numpy.random.binomial(1, p)
    # retorna a nota de cada aluno
    return notas_prova


def calcula_pr(theta_aluno, questao):
    """ calcula a probabilidade de um aluno acertar todas as questoes de uma determinada prova """
    a = questao['a']
    b = questao['b']
    coef = numpy.exp(a * (theta_aluno - b))
    return coef / (1 + coef)


def melhor_prova(tamanho_prova):
    """ Determina qual a melhor prova para o aluno 5 (a que maximiza as chances de acerto) em relacao ao aluno 4"""
    # uma 'matriz' com colunas 'index' e 'diff'
    # isso serve para preservar os indíces originais das questões depois da ordenação
    dtype = [('index', int), ('diff', float)]
    diferencas = numpy.array(numpy.zeros(100), dtype=dtype)

    melhores_questoes = numpy.zeros(tamanho_prova)

    pr_questoes_aluno4 = calcula_pr(alunos[3], tabela_questoes)
    pr_questoes_aluno5 = calcula_pr(alunos[4], tabela_questoes)

    for i in range(len(pr_questoes_aluno5)):
        diferencas[i] = (i, pr_questoes_aluno5[i] - pr_questoes_aluno4[i])

    # ordena todos pela diferença entre a probabilidade do aluno 5 e a do aluno 4
    diferencas.sort(order='diff')
    diferencas = diferencas[::-1]  # ordem descrescente

    # seleciona as n primeiras questões, pelo tamanho da prova
    for j in range(tamanho_prova):
        melhores_questoes[j] = diferencas[j]['index']

    return melhores_questoes


def saida_3_2():
    """ qual a probabilidade de o aluno 5 ser melhor que os alunos 1, 2, 3 e 4"""
    print("inicio do item 3.2")
    calcula_pr_questoes()
    saida32 = open(path+'II2.txt', 'w')
    for tamanho in [10, 20, 50, 100]:
        print("prova de tamanho: ", tamanho)
        _melhor_prova = melhor_prova(tamanho)
        pr_aluno5_melhor_prova = simula_provas(tamanho, n_simulacoes,
                                               lambda n_questoes: tabela_questoes.iloc[_melhor_prova])
        numpy.savetxt(saida32, pr_aluno5_melhor_prova.reshape(1, 4), fmt='%.5f')
    saida32.close()
    print("fim item 3.2")


def saida_3_3():
    print("entrada item 3.3")
    saida3 = open(path+'II3.txt', 'w')
    # simula a nota dos 5 alunos para cada um dos 5 alunos dadas provas aleatorias de tamanho 10, 20, 50 e 100
    calcula_pr_questoes()
    for tamanho in [10, 20, 50, 100]:
        todas_notas = pandas.DataFrame(columns=['Aluno1', 'Aluno2', 'Aluno3', 'Aluno4', 'Aluno5'])
        provas = melhor_prova(tamanho)
        provas = tabela_questoes.loc[provas]
        for i in range(n_simulacoes):
            notas_prova = calcula_notas_alunos(provas)
            todas_notas.loc[i] = notas_prova
        # calcula o intervalo de confianca para  alpha = 0,1 de cada aluno
        intervalos = []
        for aluno in range(0, 5):
            print("Aluno: %s" % aluno)
            # seleciona as notas do aluno
            notas = todas_notas.iloc[:, aluno].sort_values()
            print(notas.describe())
            print(notas)
            # com o array ordenado, excluir os primeiros e últimos 5% da amostra para obter
            # um intervalo de confiança de 90%
            inicio = notas.iloc[int(n_simulacoes * 0.05)]
            fim = notas.iloc[int(math.ceil(n_simulacoes * 0.95))]
            # adiciona o intervalo à lista de intervalos
            intervalos.append(inicio)
            intervalos.append(fim)
        # escreve a saída
        saida3.write(' '.join(map('{:.5f}'.format, intervalos)))
        saida3.write('\n')
    print("fim item 3.3")
    saida3.close()


def calcula_acertos_alunos(prova):
    """ calcula o desempenho dos alunos para a prova passada """
    notas_prova = numpy.zeros(5)
    # para cada um dos 5 alunos
    for i in xrange(0, 5):
        # calcula a probabilidade de acerto de todas as questoes da prova para o aluno i
        for p in prova['pr_' + str(i)]:
            # aplica a distribuicao binomial em cima da probabilidade de acerto para o aluno i e assim calcula sua nota
            notas_prova[i] += numpy.random.binomial(1, p)
    # retorna a nota de cada aluno
    return notas_prova


def saida_3_4():
    print("entrada item 3.4")
    saida4 = open('%sII4.txt' % path, 'w')
    calcula_pr_questoes()
    for tamanho in [10, 20, 50, 100]:
        todas_notas = pandas.DataFrame(columns=['Aluno1', 'Aluno2', 'Aluno3', 'Aluno4', 'Aluno5'])
        provas = melhor_prova(tamanho)
        provas = tabela_questoes.loc[provas]

        for i in xrange(n_simulacoes):
            notas_prova = calcula_acertos_alunos(provas)
            todas_notas.loc[i] = notas_prova

        intervalos = []
        # para cada um dos 5 alunos
        for i in range(0, 5):
            notas_aluno = todas_notas.iloc[:, i]

            media = notas_aluno.mean()
            sigma_linha = notas_aluno.std(ddof=1)

            v = stats.norm.ppf(0.95)

            inicio = media - v * sigma_linha
            fim = media + v * sigma_linha
            intervalos.append(inicio / tamanho)
            intervalos.append(fim / tamanho)

        saida4.write(' '.join(map('{:.5f}'.format, intervalos)))
        saida4.write('\n')
    print("fim item 3.4")
    saida4.close()
    return 0


def main_pt2():
    #saida_3_1()
    #saida_3_2()
    #saida_3_3()
    saida_3_4()

main_pt2()