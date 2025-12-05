import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import time
import numpy as np
import pandas as pd

plt.rc('font', size=16)  # 기본 폰트 크기
plt.rc('axes', labelsize=16)  # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=14)  # x축 눈금 폰트 크기
plt.rc('ytick', labelsize=14)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=12)  # 범례 폰트 크기
plt.rc('figure', titlesize=20)

data1 = pd.read_csv('data/Wifi-localization.csv')
data1n = data1.values.copy()
x = data1.drop('room', axis=1)
x = x.astype(float)
y = data1['room']
y = y.astype(int)
minmax = preprocessing.MinMaxScaler()
x = minmax.fit_transform(x)
onehot = preprocessing.OneHotEncoder()
y_hot = onehot.fit_transform(y.values.reshape(-1, 1)).todense()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y_hot, test_size = 0.4, random_state=0, shuffle = True)

wall_time = []
accuracy = []

algorithms = ['random_hill_climb','simulated_annealing', 'genetic_alg']

learning_rates = np.logspace(-3, 1, 8)
restarts = [0, 5, 10, 15, 20]
rhc_accuracy = 0
rhc_best_lr = 0
rhc_best_restart = 0
rhc_plot1 = []
rhc_plot2 = []
for lr in learning_rates:
    nn_rhc = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[0], max_iters=1000, bias=True, is_classifier=True, learning_rate=lr, max_attempts=100, random_state=0, curve=True, early_stopping = True, restarts=1)
    nn_rhc.fit(Xtrain, Ytrain)
    train_accuracy = accuracy_score(Ytrain, nn_rhc.predict(Xtrain))
    rhc_plot1.append(train_accuracy)
    if rhc_accuracy < train_accuracy:
        rhc_accuracy = train_accuracy
        rhc_best_lr = lr

rhc_accuracy = 0
for rs in restarts:
    nn_rhc = mlrose_hiive.NeuralNetwork(hidden_nodes=[30], activation='relu', algorithm=algorithms[0], max_iters=1000, bias=True, is_classifier=True, learning_rate=rhc_best_lr, max_attempts=100, random_state=0, curve=True, early_stopping=True, restarts=rs)
    nn_rhc.fit(Xtrain, Ytrain)
    train_accuracy = accuracy_score(Ytrain, nn_rhc.predict(Xtrain))
    rhc_plot2.append(train_accuracy)
    if rhc_accuracy < train_accuracy:
        rhc_accuracy = train_accuracy
        rhc_best_restart = rs

print('RHC :', rhc_accuracy)
print('RHC_lr :', rhc_best_lr)
print('RHC_rs :', rhc_best_restart)
plt.figure(0)
plt.semilogx(learning_rates, rhc_plot1, marker='o')
plt.title('RHC algorithm in NN (learning_rate)')
plt.grid()
plt.xlabel('learning rate')
plt.ylabel('accuracy')
plt.savefig('rhc_plot_learning_rate.png')

plt.figure(1)
plt.plot(restarts, rhc_plot2, marker='o')
plt.title('RHC algorithm in NN (restart)')
plt.grid()
plt.xlabel('number of restart')
plt.ylabel('accuracy')
plt.savefig('rhc_plot_restart.png')


learning_rates = np.logspace(-3, 1, 8)
sa_accuracy = 0
sa_best_lr = 0
sa_plot = []  # 값 모음
for lr in learning_rates:
    nn_sa = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[1], max_iters=1000, bias=True, is_classifier=True, learning_rate=lr, max_attempts=100, random_state=0, curve=True, early_stopping = True, schedule=mlrose_hiive.GeomDecay())
    nn_sa.fit(Xtrain, Ytrain)
    train_accuracy = accuracy_score(Ytrain, nn_sa.predict(Xtrain))
    sa_plot.append(train_accuracy)
    if sa_accuracy < train_accuracy:
        sa_accuracy = train_accuracy
        sa_best_lr = lr
print('SA :', sa_accuracy)
print('SA_lr :', sa_best_lr)
plt.figure(2)
plt.semilogx(learning_rates, sa_plot, marker='o')
plt.title('SA algorithm in NN (learning rate)')
plt.grid()
plt.xlabel('learning rate')
plt.ylabel('accuracy')
plt.savefig('sa_plot_learning_rate.png')

learning_rates = np.logspace(-3, 1, 8)
mutations = [0.05, 0.1, 0.15, 0.2]
pops = [200, 400, 600, 800]
ga_accuracy = 0
ga_best_lr = 0
ga_best_mutation = 0
ga_best_pop = 0
ga_plot1 = []
ga_plot2 = []
ga_plot3 = []
for lr in learning_rates:
    nn_ga = mlrose_hiive.NeuralNetwork(hidden_nodes=[30], activation='relu', algorithm=algorithms[2], max_iters=1000, bias=True, is_classifier=True, learning_rate=lr, max_attempts=100, random_state=0, curve=True, early_stopping=True, mutation_prob=0.1, pop_size=500)
    nn_ga.fit(Xtrain, Ytrain)
    train_accuracy = accuracy_score(Ytrain, nn_ga.predict(Xtrain))
    ga_plot1.append(train_accuracy)
    if ga_accuracy < train_accuracy:
        ga_accuracy = train_accuracy
        ga_best_lr = lr

ga_accuracy = 0
for mutation in mutations:
    nn_ga = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[2], max_iters=1000, bias=True, is_classifier=True, learning_rate=ga_best_lr, max_attempts=100, random_state=0, curve=True, early_stopping = True, mutation_prob=mutation, pop_size=200)
    nn_ga.fit(Xtrain, Ytrain)
    train_accuracy = accuracy_score(Ytrain, nn_ga.predict(Xtrain))
    ga_plot2.append(train_accuracy)
    if ga_accuracy < train_accuracy:
        ga_accuracy = train_accuracy
        ga_best_mutation = mutation

ga_accuracy = 0
for pop in pops:
    nn_ga = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[2], max_iters=1000, bias=True, is_classifier=True, learning_rate=ga_best_lr, max_attempts=100, random_state=0, curve=True, early_stopping = True, mutation_prob=ga_best_mutation, pop_size=pop)
    nn_ga.fit(Xtrain, Ytrain)
    train_accuracy = accuracy_score(Ytrain, nn_ga.predict(Xtrain))
    ga_plot3.append(train_accuracy)
    if ga_accuracy < train_accuracy:
        ga_accuracy = train_accuracy
        ga_best_pop = pop

print('GA :', ga_accuracy)
print('GA_lr :', ga_best_lr)
print('GA_mutation :', ga_best_mutation)
print('GA_pop :', ga_best_pop)

plt.figure(3)
plt.semilogx(learning_rates, ga_plot1, marker='o')
plt.title('GA algorithm in NN (learning rate)')
plt.grid()
plt.xlabel('learning rates')
plt.ylabel('accuracy')
plt.savefig('ga_plot_lr.png')

plt.figure(4)
plt.plot(mutations, ga_plot2, marker='o')
plt.title('GA algorithm in NN (mutation)')
plt.grid()
plt.xlabel('mutation probs')
plt.ylabel('accuracy')
plt.savefig('ga_plot_mutation.png')

plt.figure(5)
plt.plot(pops, ga_plot3, marker='o')
plt.title('GA algorithm in NN (population)')
plt.grid()
plt.xlabel('populations')
plt.ylabel('accuracy')
plt.savefig('ga_plot_pop.png')

#Plots for accuracy and time
wall_time = []
best_nn_rhs = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[0], max_iters=1000, bias=True, is_classifier=True, learning_rate=rhc_best_lr, max_attempts=100, random_state=0, curve=True, early_stopping = True, restarts=rhc_best_restart)
best_nn_sa = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[1], max_iters=1000, bias=True, is_classifier=True, learning_rate=sa_best_lr, max_attempts=100, random_state=0, curve=True, early_stopping = True, schedule=mlrose_hiive.GeomDecay())
best_nn_ga = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[2], max_iters=1000, bias=True, is_classifier=True, learning_rate=ga_best_lr, max_attempts=100, random_state=0, curve=True, early_stopping = True, mutation_prob=ga_best_mutation, pop_size=ga_best_pop)
best_nn_hw1 = MLPClassifier(alpha=0.001, hidden_layer_sizes=30, random_state=0, max_iter=1000, learning_rate_init= 0.00278, solver='adam')

start_time = time.time()
best_nn_rhs.fit(Xtrain, Ytrain)
end_time = time.time()
wall_time.append(end_time-start_time)
print('RHS time:', end_time-start_time)

start_time = time.time()
best_nn_sa.fit(Xtrain, Ytrain)
end_time = time.time()
wall_time.append(end_time-start_time)
print('SA time:', end_time-start_time)

start_time = time.time()
best_nn_ga.fit(Xtrain, Ytrain)
end_time = time.time()
wall_time.append(end_time-start_time)
print('GA time:', end_time-start_time)

start_time = time.time()
best_nn_hw1.fit(Xtrain, Ytrain)
end_time = time.time()
wall_time.append(end_time-start_time)
print('NN_hw1 time:', end_time-start_time)

RHS_Ypred = best_nn_rhs.predict(Xtest)
RHS_test_accuracy = accuracy_score(Ytest, RHS_Ypred)
SA_Ypred = best_nn_sa.predict(Xtest)
SA_test_accuracy = accuracy_score(Ytest, SA_Ypred)
GA_Ypred = best_nn_ga.predict(Xtest)
GA_test_accuracy = accuracy_score(Ytest, GA_Ypred)
HW1_Ypred = best_nn_hw1.predict(Xtest)
HW1_test_accuracy = accuracy_score(Ytest, HW1_Ypred)

print('RHS accuracy:', RHS_test_accuracy)
print('SA accuracy:', SA_test_accuracy)
print('GA accuracy:', GA_test_accuracy)
print('NN_hw1 accuracy:', HW1_test_accuracy)

plt.figure(6)
als = ['RHC', 'SA', 'GA', 'Back prop']
x = np.arange(len(als))
bar = plt.bar(x, wall_time)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.4f'%height, ha='center', va='bottom', size=12)
plt.title('Training Times for four algorithms')
plt.xticks(x, als)
plt.ylabel('Training Time (seconds)')
plt.tight_layout()
plt.savefig('Compare_time.png')

plt.figure(7)
best_score = [RHS_test_accuracy, SA_test_accuracy, GA_test_accuracy, HW1_test_accuracy]
als = ['RHC', 'SA', 'GA', 'Backprop']
x = np.arange(len(als))
bar = plt.bar(x, best_score)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.4f'%height, ha='center', va='bottom', size=12)
plt.title('Accuracy for four algorithms')
plt.xticks(x, als)
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('Compare_accuracy.png')

best_nn_rhs = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[0], max_iters=1000, bias=True, is_classifier=True, learning_rate=rhc_best_lr, max_attempts=100, random_state=0, curve=True, early_stopping = True)
best_nn_sa = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[1], max_iters=1000, bias=True, is_classifier=True, learning_rate=sa_best_lr, max_attempts=100, random_state=0, curve=True, early_stopping = True, schedule=mlrose_hiive.GeomDecay())
best_nn_ga = mlrose_hiive.NeuralNetwork(hidden_nodes = [30], activation='relu', algorithm=algorithms[2], max_iters=1000, bias=True, is_classifier=True, learning_rate=ga_best_lr, max_attempts=100, random_state=0, curve=True, early_stopping = True, mutation_prob=ga_best_mutation, pop_size=ga_best_pop)
best_nn_hw1 = MLPClassifier(alpha=0.001, hidden_layer_sizes=30, random_state=0, max_iter=1000, learning_rate_init= 0.00278, solver='adam', warm_start=True)

best_nn_rhs.fit(Xtrain, Ytrain)
best_nn_sa.fit(Xtrain, Ytrain)
best_nn_ga.fit(Xtrain, Ytrain)
best_nn_hw1.fit(Xtrain, Ytrain)

plt.figure(8)
plt.plot(best_nn_rhs.fitness_curve[:,0])
print('no of fitness_rhc:', np.size(best_nn_rhs.fitness_curve[:,0]))
plt.title("Loss function of RHS")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('rhs-loss.png')

plt.figure(9)
plt.plot(best_nn_sa.fitness_curve[:,0])
print('no of fitness_sa:', np.size(best_nn_sa.fitness_curve[:,0]))
plt.title("Loss function of SA")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('sa-loss.png')

plt.figure(10)
plt.plot(best_nn_ga.fitness_curve[:,0])
print('no of fitness_ga:', np.size(best_nn_ga.fitness_curve[:,0]))
plt.title("Loss function of GA")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('ga-loss.png')

plt.figure(11)
plt.plot(best_nn_hw1.loss_curve_)
print('no of fitness_nn:', np.size(best_nn_hw1.loss_curve_))
plt.title("Loss function of BackProp")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('hw1-loss.png')