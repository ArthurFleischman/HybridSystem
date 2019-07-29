import numpy as np
import sys
from sklearn import metrics, svm


def pso(n_particles, iterations, trainset, traintarget, valset, valtarget):
    dimension = 3
    # inicializa partículas aleatoriamente
    x = np.random.rand(n_particles, dimension)
    C_boundMax = 10000  # limite superior para o C
    C_boundMin = 0.01  # limite inferior para o C
    v_cMax = 100  # velocidade máxima para o C
    v_cMin = -100  # velocidade mínima para o C

    epsilonBoundMax = 0.1  # limite superior para o epsilon
    epsilonBoundMin = 0.00000001  # limite inferior para o epsilon
    v_eps_Max = 100  # velocidade máxima para o epsilon
    v_eps_Min = -10  # velocidade mínima para o epsilon

    gammaBoundMax = 1000  # limite superior para o gamma
    gammaBoundMin = 0.01  # limite inferior para o gamma
    v_gammaMax = 50  # velocidade máxima para o gamma
    v_gammaMin = -50  # velocidade mínima para o gamma

    # normaliza as particulas dentro do range de C
    x[:, 0] = x[:, 0]*(C_boundMax-C_boundMin)+C_boundMin
    # normaliza as particulas dentro do range de gamma
    x[:, 1] = x[:, 1]*(gammaBoundMax-gammaBoundMin) + gammaBoundMin
    # normaliza as particulas dentro do range de epsilon
    x[:, 2] = x[:, 2]*(epsilonBoundMax-epsilonBoundMin) + epsilonBoundMin

    x_val = np.zeros(n_particles)

    p_best = np.zeros((n_particles, dimension))  # inicializando personal best
    # inicializando p_best_val
    p_best_val = np.zeros(n_particles) + sys.maxsize

    v = np.zeros((n_particles, dimension))  # inicializando velocidades

    g_best = np.zeros(dimension)  # inicializar melhor indivíduo
    g_best_val = 0 + sys.maxsize  # inicializar melhor valor

    best_iteration = np.zeros(iterations)
    C_1 = 2
    C_2 = 2
    w = 1

    for i in range(iterations):

        for j in range(n_particles):

            x_val[j] = fitness(x[j], trainset, traintarget, valset, valtarget)

            if(x_val[j] < p_best_val[j]):  # avaliando p_best
                p_best_val[j] = x_val[j]
                p_best[j] = x[j].copy()

        min_index = np.argmin(p_best_val)

        if(p_best_val[min_index] < g_best_val):  # avaliando g_best
            g_best_val = p_best_val[min_index]
            g_best = p_best[min_index].copy()

        w = 1/(i + 1)

        for j in range(n_particles):

            rand_1 = np.random.random()
            rand_2 = np.random.random()

            v[j] = w*v[j] + C_1*(p_best[j]-x[j])*rand_1 + \
                C_2*(g_best - x[j])*rand_2
            #v[j] = v[j] + C_1*(p_best[j]-x[j])*rand_1 + C_2*(g_best - x[j])*rand_2
            x[j] = x[j] + v[j]

            # Checar limites dos parâmetros
            x[j, 2] = boundCheck(x[j, 2], epsilonBoundMax, epsilonBoundMin)
            x[j, 1] = boundCheck(x[j, 1], gammaBoundMax, gammaBoundMin)
            x[j, 0] = boundCheck(x[j, 0], C_boundMax, C_boundMin)

            v[j, 2] = boundCheck(v[j, 2], v_eps_Max, v_eps_Min)
            v[j, 1] = boundCheck(v[j, 1], v_gammaMax, v_gammaMin)
            v[j, 0] = boundCheck(v[j, 0], v_cMax, v_cMin)

        best_iteration[i] = g_best_val

    return g_best


def fitness(x, trainset, traintarget, valset, valtarget):

    model = svm.SVR(C=x[0], gamma=x[1], epsilon=x[2])
    model.fit(trainset, traintarget)
    predicts = model.predict(valset)
    erro = metrics.mean_squared_error(valtarget, predicts)

    return erro


def boundCheck(val, upperbound, lowerbound):
    resp = val
    if resp < lowerbound:
        resp = lowerbound
    if resp > upperbound:
        resp = upperbound
    return resp
