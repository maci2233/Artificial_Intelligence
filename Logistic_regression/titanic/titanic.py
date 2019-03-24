from csv import reader
import math
import matplotlib.pyplot as plt

'''
PROBAR CON Y SIN FEATURE Fare
PROBAR PONIENDO MISSING VALUES COMO NONE Y COMO LA MEDIA DEL RESTO
'''

#Open the csv file and return it as a list
def load_csv(filename):
    dataset = list()
    with open(filename, "r") as file:
        rows = reader(file)
        for row in rows:
            if not row:
                continue
            dataset.append(row)
    return dataset

#Modified map function to that the func passed as parameter is only applied
#to the indexes that are in the ind_list
def map_iter(func, row, ind_list):
    for ind in ind_list:
        if row[ind] != '':
            row[ind] = func(row[ind])

#Prepare the dataset to get rid of the non-required information
#and separate the features from the labels
#There are 2 ways of preparing the data, if  train=True then we get all the sample attributes and the class
#If train = False that means we are testing, so we get all the sample attributes and the PassengerId.
def prepare_titanic_dataset(dataset, train):
    n = len(dataset)
    if train:
        for i in range(1, n):
            dataset[i][4] = 0 if dataset[i][4] == 'male' else 1 if dataset[i][4] == 'female' else ''
            map_iter(int, dataset[i], [1,2,6,7])
            map_iter(float, dataset[i], [5])
            dataset[i] = dataset[i][1:3] + dataset[i][4:8]
    else:
        for i in range(1, n):
            dataset[i][3] = 0 if dataset[i][3] == 'male' else 1 if dataset[i][3] == 'female' else ''
            map_iter(int, dataset[i], [1,5,6])
            map_iter(float, dataset[i], [4])
            dataset[i] = dataset[i][:2] + dataset[i][3:7]
    return dataset[1:]

#To make predictions by first solving the linear function, when we get y which is our hypothesis
#We use the sigmoid function to get a value strictly from 0 to 1 in order to make the classification
def predict(sample, params):
    yh = params[0]
    for i in range(len(sample)):
        if sample[i] != '':
            yh += params[i+1] * sample[i]
    return 1.0 / (1.0 + math.exp(-yh))

#In order to update the params we iterate through all samples and all attributes
#params[0] which is the bias is not multipled by any atribute because it doesn´t affect them directly
def sgd(params, samples, y, l_rate):
    for i in range(len(samples)):
        yh = predict(samples[i], params)
        error = y[i] - yh
        params[0] = params[0] + l_rate * error * yh * (1.0 - yh)
        for j in range(len(samples[i])):
            if samples[i][j] != '':
                params[j+1] = params[j+1] + l_rate * error * yh * (1.0 - yh) * samples[i][j]
    return params

#This function is used to get 2 important facts, the accumulated error of all samples
#and also the accuracy of the model because it rounds each prediction and compares it
# with the real class, so that way we can know how many predictions where correct.
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

#Graph all the acummulated errors and accuracy over time
def graph_info(error_list, precision_list):
    plt.plot(precision_list)
    plt.plot(error_list)
    plt.show()

#This is the core function to train the model, for a specific amount of epochs, sgd function
#will be called to update the parameters to make better predictions, each time the parameters
#are obtained we get the error and accuracy to graph them, we return the final parameters at the end
def logistic_regression_train(samples, y, params):
    errors, precision = list(), list()
    l_rate = 0.0001
    epochs = 1
    while True:
        old_params = list(params)
        params = sgd(params, samples, y, l_rate)
        err = get_error(params, samples, y)
        errors.append(err[0])
        precision.append(err[1])
        if old_params == params or epochs == 3000:
            print('Final parameters')
            print(params)
            print('Final error with decimal predictions:')
            print(errors[-1])
            print('Precision with rounded predictions:')
            print(precision[-1])
            graph_info(errors, precision)
            break
        epochs += 1
    return params

#this function is used to test our model, so it returns the class predictions that it made for each sample
def titanic_test(samples, params):
    classifications = list()
    for sample in samples:
        yh = round(predict(sample, params))
        classifications.append(yh)
    return classifications

#Kaggle expects a csv file with PassengerId, class in order to evaluate our model
def generate_csv(ids, classifications):
    with open("results.csv", "w") as file_csv:
        print('PassengerId,Survived', file=file_csv)
        for i in range(len(classifications)):
            print(str(ids[i]) + ',' + str(classifications[i]), file=file_csv)

#Main function, here we get the training dataset, prepare it and train our model.
#Then we use the final parameters to test our model with the test dataset
#At the end a csv file is generated to upload it to kaggle
if __name__ == '__main__':
    filename = "train.csv"
    train_dataset = prepare_titanic_dataset(load_csv(filename), True)
    train_samples, train_y = list(), list()
    for row in train_dataset:
        train_samples.append(row[1:])
        train_y.append(row[0])
    params = [0.0] + [0.0 for i in range(len(train_samples[0]))]
    final_params = logistic_regression_train(train_samples, train_y, params)
    filename = "test.csv"
    test_dataset = prepare_titanic_dataset(load_csv(filename), False)
    test_samples = list()
    test_ids = list()
    for row in test_dataset:
        test_samples.append(row[1:])
        test_ids.append(row[0])
    test_classifications = titanic_test(test_samples, final_params)
    generate_csv(test_ids, test_classifications)
