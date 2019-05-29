# -*- coding: utf-8 -*-
# Bianca L. Santos, Cleiton Dantas, Fábio Oliveira , Mariana Santana
import math
import numpy
import pandas


""" numero de simulações a serem feitas"""
n_simulacoes = 100000

""" habilidades dos alunos """
alunos = [-1.0, -0.5, 0.0, 0.5, 1.0]

"""importa as questoes do arquivo passado pelo prof """
tabela_questoes = pandas.read_table('../dados/questoes.txt', sep=' ', names=('a', 'b'))


def gera_provas_aleatorias(n_questoes):
    """ gera provas aleatorias dado o numero de questoes que deseja pra prova e a tabela de questoes """
    # reindexa a tabela de prova e permita randomicamente as linhas , não destrutivo
    prova_aleatoria = tabela_questoes.reindex(numpy.random.permutation(tabela_questoes.index))
    # separo de todas as linhas embaralhadas randomicamente o numero de questoes que me interessa
    prova_aleatoria = prova_aleatoria.head(n_questoes)
    return prova_aleatoria


def calcula_pr(theta_aluno, questao):
    """ calcula a probabilidade de um aluno acertar todas as questoes de uma determinada prova """
    a = questao['a']
    b = questao['b']
    coef = numpy.exp(a * (theta_aluno - b))
    return coef / (1 + coef)


def calcula_pr_questoes():
    for i in xrange(0, 5):
        tabela_questoes['pr_' + str(i)] = calcula_pr(alunos[i], tabela_questoes)


def calcula_notas_alunos(prova):
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


def compara_aluno5(notas_prova, vezes_aluno5_melhor):
    """Compara a nota do aluno 5 com os demais alunos em casa uma das provas e retorna """
    nota_aluno5 = notas_prova[4]
    # soma a quantidade de vezes que o aluno 5 foi melhor do que os demais
    for i in xrange(0, 4):
        if nota_aluno5 > notas_prova[i]:
            vezes_aluno5_melhor[i] += 1
    return vezes_aluno5_melhor


def simula_provas(tamanho_prova, quantidade_simulacoes, gerador_provas):
    """ Efetivamente entrar no problema:
    Chama as funções necessárias para dar a probabilidade do aluno 5 ser melhor que os demais """
    # habilidades de cada um dos 5 alunos
    vezes_aluno5_melhor = numpy.zeros(4)

    # para um numero alto de vezes, verifica quantas vezes o aluno 5 foi melhor em relação aos demais
    for i in xrange(quantidade_simulacoes):
        prova_aleatoria = gerador_provas(tamanho_prova)
        notas_prova = calcula_notas_alunos(prova_aleatoria)
        compara_aluno5(notas_prova, vezes_aluno5_melhor)

    probabilidade_aluno5_melhor = vezes_aluno5_melhor / quantidade_simulacoes
    return probabilidade_aluno5_melhor


def melhor_prova(tamanho_prova):
    """ Determina qual a melhor prova para o aluno 5 (a que maximiza as chances de acerto) em relacao ao aluno 4"""

    # uma 'matriz' com colunas 'index' e 'diff'
    # isso serve para preservar os indíces originais das questões depois da ordenação
    dtype = [('index', int), ('diff', float)]
    diferencas = numpy.array(numpy.zeros(100), dtype=dtype)

    melhores_questoes = numpy.zeros(tamanho_prova)

    pr_questoes_aluno4 = calcula_pr(alunos[3], tabela_questoes)
    pr_questoes_aluno5 = calcula_pr(alunos[4], tabela_questoes)

    for i in xrange(len(pr_questoes_aluno5)):
        diferencas[i] = (i, pr_questoes_aluno5[i] - pr_questoes_aluno4[i])

    # ordena todos pela diferença entre a probabilidade do aluno 5 e a do aluno 4
    diferencas.sort(order='diff')
    diferencas = diferencas[::-1]  # ordem descrescente

    # seleciona as n primeiras questões, pelo tamanho da prova
    for j in xrange(tamanho_prova):
        melhores_questoes[j] = diferencas[j]['index']

    return melhores_questoes


def primeiro_item():
    # chamadas para resolucao o primeiro item do ep
    pr_aluno5_melhor_p10 = simula_provas(10, n_simulacoes, gera_provas_aleatorias)
    pr_aluno5_melhor_p20 = simula_provas(20, n_simulacoes, gera_provas_aleatorias)
    pr_aluno5_melhor_p50 = simula_provas(50, n_simulacoes, gera_provas_aleatorias)
    pr_aluno5_melhor_p100 = simula_provas(100, n_simulacoes, gera_provas_aleatorias)

    # compara desempenho do aluno 5 com demais alunos
    comparacao_aluno5_todas_provas = pandas.DataFrame(columns=['Aluno1', 'Aluno2', 'Aluno3', 'Aluno4'])
    comparacao_aluno5_todas_provas.loc[0] = pr_aluno5_melhor_p10
    comparacao_aluno5_todas_provas.loc[1] = pr_aluno5_melhor_p20
    comparacao_aluno5_todas_provas.loc[3] = pr_aluno5_melhor_p50
    comparacao_aluno5_todas_provas.loc[4] = pr_aluno5_melhor_p100

    # cria o arquivo final com a comparacao do aluno 5 para os demais na pasta dados
    numpy.savetxt(r'../dados/I1.txt', comparacao_aluno5_todas_provas.values, fmt='%.5f')


def segundo_item():
    saida2 = open('../dados/I2.txt', 'w')

    # chamadas para resolucao do segundo item do ep
    for tamanho in [10, 20, 50]:
        _melhor_prova = melhor_prova(tamanho)
        numpy.savetxt(saida2, _melhor_prova.reshape(1, tamanho), fmt='%d')
        pr_aluno5_melhor_prova = simula_provas(tamanho, n_simulacoes,
                                               lambda n_questoes: tabela_questoes.iloc[_melhor_prova])
        numpy.savetxt(saida2, pr_aluno5_melhor_prova.reshape(1, 4), fmt='%.5f')

    saida2.close()


def terceiro_item():
    saida3 = open('../dados/I3.txt', 'w')

    # simula a nota dos 5 alunos para cada um dos 5 alunos dadas provas aleatorias de tamanho 10, 20, 50 e 100
    for tamanho in [10, 20, 50, 100]:
        todas_notas = pandas.DataFrame(columns=['Aluno1', 'Aluno2', 'Aluno3', 'Aluno4', 'Aluno5'])
        provas = melhor_prova(tamanho)
        provas = tabela_questoes.loc[provas]
        for i in xrange(n_simulacoes):
            notas_prova = calcula_notas_alunos(provas)
            todas_notas.loc[i] = notas_prova

        # calcula o intervalo de confianca para  alpha = 0,1 de cada aluno
        intervalos = []

        for aluno in range(0, 5):
            # seleciona as notas do aluno
            notas = todas_notas.iloc[:, aluno].sort_values()
            # com o array ordenado, excluir os primeiros e últimos 5% da amostra para obter
            # um intervalo de confiança de 90%
            inicio = notas.iloc[int(n_simulacoes * 0.05)]
            fim = notas.iloc[int(math.ceil(n_simulacoes * 0.95))]

            # adiciona o intervalo à lista de intervalos
            intervalos.append((inicio/tamanho))
            intervalos.append((fim/tamanho))

        # escreve a saída
        saida3.write(' '.join(map('{:.5f}'.format, intervalos)))
        saida3.write('\n')

    saida3.close()


def main():
    calcula_pr_questoes()
    """ Chama a função responsável pelas simulações para todos os N tamanhos de provas diferentes """

    print("Primeiro Item")
    primeiro_item()

    print("Segundo Item")
    segundo_item()

    print("Terceiro Item")
    terceiro_item()


main()
