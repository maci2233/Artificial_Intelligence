import math
import matplotlib.pyplot as plt


def predict(sample, params):
    yh = params[0]
    for i in range(len(sample)):
        yh += params[i+1] * sample[i]
    return 1.0 / (1.0 + math.exp(-yh))


def sgd(params, samples, y, l_rate):
    for i in range(len(samples)):
        yh = predict(samples[i], params)
        error = y[i] - yh
        params[0] = params[0] + l_rate * error * yh * (1.0 - yh)
        for j in range(len(samples[i])):
            params[j+1] = params[j+1] + l_rate * error * yh * (1.0 - yh) * samples[i][j]
    return params


def get_error(params, samples, y):
    error_acum = 0
    corr_pred = 0
    n = len(samples)
    for i in range(n):
        yh = predict(samples[i], params)
        yhr = round(yh)
        if yhr == y[i]:
            corr_pred += 1
        if y[i] == 1:
            if yh == 0:
                yh = .00001;
            error = (-1)*math.log(yh);
        if y[i] == 0:
            if yh == 1:
                yh = .99999;
            error = (-1)*math.log(1-yh);
        error_acum += error
    return [(error_acum / n) * 100, (corr_pred / n) * 100]


def graph_info(errors, precision):
    plt.plot(precision)
    plt.plot(errors)
    plt.show()


def logistic_regression_train(samples, y, params):
    errors, precision = list(), list()
    l_rate = 0.03
    epochs = 1
    while True:
        old_params = list(params)
        params = sgd(params, samples, y, l_rate)
        err = get_error(params, samples, y)
        errors.append(err[0])
        precision.append(err[1])
        if old_params == params or epochs == 1500:
            print('Final parameters')
            print(params)
            print('Final error with decimal predictions:')
            print(errors[-1])
            print('Precision with rounded predictions:')
            print(precision[-1])
            #graph_info(errors, precision)
        epochs += 1
    return params


def logistic_regression_test(samples, y, params):
    precision = list()
    corr_pred = 0
    for i in range(len(samples)):
        yh = predict(sample[i], params)
        yh = round(yh)
        if yh == y[i]:
            corr_pred += 1
    precision = corr_pred / len(samples)
    return precision


def main():
    train_samples = [[12],[23],[15],[70],[85],[7],[25]] #value of x for each sample(x represents an age)
    train_y = [0,1,0,1,1,0,1] #Classification for each age in the sample, 0 = underage and 1 = adult
    params = [0.0]+[0.0 for i in range(len(train_samples[0]))]  #first params is the bias, the rest are the parameters for each x, there is only 1 parameter because each sample has only one attribute
    final_params = logistic_regression_train(train_samples, train_y, params)
    test_samples = list()
    test_y = list()
    for i in range(1, 100):
        test_samples.append([i])
    for i in range(1, 19):
        test_y.append(0)
    for i in range(19, 100):
        test_y.append(1)
    precision = logistic_regression_test(test_samples, test_y, final_params)
    print('final precision: %s', precision)


main()
