from sklearn import svm
from sklearn.metrics import mean_squared_error as MSE


def gridSVR(x_train, y_train, x_val, y_val):
    c_r = list(range(-5, 6))
    eps_r = list(range(-5, 6))
    gama_r = list(range(-5, 6))

    best_error = 999999999999999999999999999

    for c in c_r:
        for e in eps_r:
            for g in gama_r:
                model = svm.SVR(C=10**c, gamma=10**g, epsilon=10**e)
                model.fit(x_train, y_train)
                predicts = model.predict(x_val)
                error = MSE(y_val, predicts)
                if error < best_error:
                    best_error = error
                    best_model = model
                    best_predicts = predicts
                    best_param = (c, g, e)
    return (best_model, best_predicts, best_error, best_param)
