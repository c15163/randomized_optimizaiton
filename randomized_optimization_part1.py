import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rc('font', size=14)  # 기본 폰트 크기
plt.rc('axes', labelsize=14)  # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=14)  # x축 눈금 폰트 크기
plt.rc('ytick', labelsize=14)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=12)  # 범례 폰트 크기
plt.rc('figure', titlesize=18)

def queens_max(state):
    fitness_cnt = 0
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            if (state[j] != state[i]) and (state[j] != state[i] + (j - i)) and (state[j] != state[i] - (j - i)):
            # If no attacks, then increment counter
                fitness_cnt += 1
    return fitness_cnt

fitness = [mlrose_hiive.FlipFlop(), mlrose_hiive.CustomFitness(queens_max), mlrose_hiive.FourPeaks(t_pct=0.1)]
schedules = {'Exp Decay': mlrose_hiive.ExpDecay(), 'Arith Decay': mlrose_hiive.ArithDecay(), 'Geom Decay': mlrose_hiive.GeomDecay()}

def rhc_plot(problem, restart, max_iter):
    _, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem = problem, restarts=restart, max_attempts = 100, max_iters = max_iter, random_state = 0, curve=True)  # restart
    return rhc_best_fitness, rhc_fitness_curve

def sa_plot(problem, schedule, max_iter):
    _, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(problem = problem, schedule=schedule, max_attempts = 100, max_iters = max_iter, random_state = 0, curve=True)  # restart
    return sa_best_fitness, sa_fitness_curve

def ga_plot(problem, mutation_prob, pop_size, max_iter):
    _, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(problem = problem, mutation_prob=mutation_prob, pop_size=pop_size, max_attempts = 100, max_iters = max_iter, random_state = 0, curve=True)  # restart
    return ga_best_fitness, ga_fitness_curve

def mimic_plot(problem, keep_pct, pop_size, max_iter):
    _, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(problem = problem, keep_pct=keep_pct, pop_size=pop_size, max_attempts = 100, max_iters = max_iter, random_state = 0, curve=True)  # restart
    return mimic_best_fitness, mimic_fitness_curve

best_params = [[0 for col in range(6)] for row in range(3)]

#1. Filp Flop
N=100
problem = mlrose_hiive.DiscreteOpt(length = N, fitness_fn = fitness[0], maximize = True, max_val = 2)
max_iter=1000
plt.figure(0)
k = []
increment=5
for restart in range(0, 30, increment):
    rhc_fit, rhc_curve = rhc_plot(problem, restart, max_iter)
    k.append(rhc_fit)
    plt.plot(rhc_curve[:,0], label='restart: {}'.format(restart))
best_restart = np.arange(0, 30, increment)[np.argmax(k)]
best_params[0][0] = best_restart
print('The best restart is :', best_restart)
plt.legend()
plt.grid()
plt.title('Random Hill Climb')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('rhc_restart_flipflop.png')

plt.figure(1)
k=[]
for key, value in schedules.items():
    sa_fit, sa_curve = sa_plot(problem, value, max_iter)
    k.append(sa_fit)
    plt.plot(sa_curve[:,0], label='schedule: {}'.format(key))
schedules_list = list(schedules.values())
best_schedule = schedules_list[np.argmax(k)]
best_params[0][1] = best_schedule
print('The best schedule is :', best_schedule)
plt.legend()
plt.grid()
plt.title('Simulated Annealing')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('sa_schedule_flipflop.png')

plt.figure(2)
k=[]
pop_size=100
mutation = [0.1, 0.15, 0.2, 0.25, 0.3]
for mutation_prob in mutation:
    ga_fit, ga_curve = ga_plot(problem, mutation_prob, pop_size, max_iter)
    k.append(ga_fit)
    plt.plot(ga_curve[:,0], label='mutation: {}'.format(mutation_prob))
best_mutation = mutation[np.argmax(k)]
best_params[0][2] = best_mutation
print('The best mutation prob is :', best_mutation)
plt.legend()
plt.grid()
plt.title('Mutation prob for Genetic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('ga_mutation_flipflop.png')

plt.figure(3)
k=[]
mutation_prob = best_mutation
pop_sizes=[200, 400, 600, 800]
for pop_size in pop_sizes:
    ga_fit, ga_curve = ga_plot(problem, mutation_prob, pop_size, max_iter)
    k.append(ga_fit)
    plt.plot(ga_curve[:,0], label='pop: {}'.format(pop_size))
best_pop = pop_sizes[np.argmax(k)]
best_params[0][3] = best_pop
print('The best pop is :', best_pop)
plt.legend()
plt.grid()
plt.title('Pop size for Genetic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('ga_pop_flipflop.png')

plt.figure(4)
k=[]
pop_size=100
keep_pcts = [0.1, 0.2, 0.3]
for keep_pct in keep_pcts:
    mimic_fit, mimic_curve = mimic_plot(problem, keep_pct, pop_size, max_iter)
    k.append(mimic_fit)
    plt.plot(mimic_curve[:,0], label='keep_pct: {}'.format(keep_pct))
best_keep_pct = keep_pcts[np.argmax(k)]
best_params[0][4] = best_keep_pct
print('The best keep pct is :', best_keep_pct)
plt.legend()
plt.grid()
plt.title('Keep pct for Mimic algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('mimic_keepct_flipflop.png')
plt.figure(5)
k=[]
keep_pct = best_keep_pct
pop_sizes=[200, 400, 600, 800]
for pop_size in pop_sizes:
    mimic_fit, mimic_curve = mimic_plot(problem, keep_pct, pop_size, max_iter)
    k.append(mimic_fit)
    plt.plot(mimic_curve[:,0], label='pop: {}'.format(pop_size))
best_pop = pop_sizes[np.argmax(k)]
best_params[0][5] = best_pop
print('The best pop is :', best_pop)
plt.legend()
plt.grid()
plt.title('Pop size for Mimic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('mimic_pop_flipflop.png')

# Queens
N=50  # total: 1225
problem = mlrose_hiive.DiscreteOpt(length = N, fitness_fn = fitness[1], maximize = True, max_val = N)
max_iter=1000
plt.figure(6)
k = []
increment=5
for restart in range(0, 30, increment):
    rhc_fit, rhc_curve = rhc_plot(problem, restart, max_iter)
    k.append(rhc_fit)
    plt.plot(rhc_curve[:,0], label='restart: {}'.format(restart))
best_restart = np.arange(0, 30, increment)[np.argmax(k)]
best_params[1][0] = best_restart
print('The best restart is :', best_restart)
plt.legend()
plt.grid()
plt.title('Random Hill Climb')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('rhc_restart_queen.png')

plt.figure(7)
k=[]
for key, value in schedules.items():
    sa_fit, sa_curve = sa_plot(problem, value, max_iter)
    k.append(sa_fit)
    plt.plot(sa_curve[:,0], label='schedule: {}'.format(key))
schedules_list = list(schedules.values())
best_schedule = schedules_list[np.argmax(k)]
best_params[1][1] = best_schedule
print('The best schedule is :', best_schedule)
plt.legend()
plt.grid()
plt.title('Simulated Annealing')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('sa_schedule_queen.png')

plt.figure(8)
k=[]
pop_size=100
mutation = [0.1, 0.15, 0.2, 0.25, 0.3]
for mutation_prob in mutation:
    ga_fit, ga_curve = ga_plot(problem, mutation_prob, pop_size, max_iter)
    k.append(ga_fit)
    plt.plot(ga_curve[:,0], label='mutation: {}'.format(mutation_prob))
best_mutation = mutation[np.argmax(k)]
best_params[1][2] = best_mutation
print('The best mutation prob is :', best_mutation)
plt.legend()
plt.grid()
plt.title('Mutation prob for Genetic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('ga_mutation_queen.png')

plt.figure(9)
k=[]
mutation_prob = best_mutation
pop_sizes=[200, 400, 600, 800]
for pop_size in pop_sizes:
    ga_fit, ga_curve = ga_plot(problem, mutation_prob, pop_size, max_iter)
    k.append(ga_fit)
    plt.plot(ga_curve[:,0], label='pop: {}'.format(pop_size))
best_pop = pop_sizes[np.argmax(k)]
best_params[1][3] = best_pop
print('The best pop is :', best_pop)
plt.legend()
plt.grid()
plt.title('Pop size for Genetic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('ga_pop_queen.png')

plt.figure(10)
k=[]
pop_size=100
keep_pcts = [0.1, 0.2, 0.3]
for keep_pct in keep_pcts:
    mimic_fit, mimic_curve = mimic_plot(problem, keep_pct, pop_size, max_iter)
    k.append(mimic_fit)
    plt.plot(mimic_curve[:,0], label='keep_pct: {}'.format(keep_pct))
best_keep_pct = keep_pcts[np.argmax(k)]
best_params[1][4] = best_keep_pct
print('The best keep pct is :', best_keep_pct)
plt.legend()
plt.grid()
plt.title('Keep pct for Mimic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('mimic_keepct_queen.png')

plt.figure(11)
k=[]
keep_pct = best_keep_pct
pop_sizes=[200, 400, 600, 800]
for pop_size in pop_sizes:
    mimic_fit, mimic_curve = mimic_plot(problem, keep_pct, pop_size, max_iter)
    k.append(mimic_fit)
    plt.plot(mimic_curve[:,0], label='pop: {}'.format(pop_size))
best_pop = pop_sizes[np.argmax(k)]
best_params[1][5] = best_pop
print('The best pop is :', best_pop)
plt.legend()
plt.grid()
plt.title('Pop size for Mimic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('mimic_pop_queen.png')

# 4 peaks
N=100
problem = mlrose_hiive.DiscreteOpt(length = N, fitness_fn = fitness[2], maximize = True, max_val = 2)
max_iter=1000
plt.figure(12)
k = []
increment=5
for restart in range(0, 30, increment):
    rhc_fit, rhc_curve = rhc_plot(problem, restart, max_iter)
    k.append(rhc_fit)
    plt.plot(rhc_curve[:,0], label='restart: {}'.format(restart))
best_restart = np.arange(0, 30, increment)[np.argmax(k)]
best_params[2][0]= best_restart
print('The best restart is :', best_restart)
plt.legend()
plt.grid()
plt.title('Random Hill Climb')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('rhc_restart_peak.png')

plt.figure(13)
k=[]
for key, value in schedules.items():
    sa_fit, sa_curve = sa_plot(problem, value, max_iter)
    k.append(sa_fit)
    plt.plot(sa_curve[:,0], label='schedule: {}'.format(key))
schedules_list = list(schedules.values())
best_schedule = schedules_list[np.argmax(k)]
best_params[2][1] = best_schedule
print('The best schedule is :', best_schedule)
plt.legend()
plt.grid()
plt.title('Simulated Annealing')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('sa_schedule_peak.png')

plt.figure(14)
k=[]
pop_size=100
mutation = [0.1, 0.15, 0.2, 0.25, 0.3]
for mutation_prob in mutation:
    ga_fit, ga_curve = ga_plot(problem, mutation_prob, pop_size, max_iter)
    k.append(ga_fit)
    plt.plot(ga_curve[:,0], label='mutation: {}'.format(mutation_prob))
best_mutation = mutation[np.argmax(k)]
best_params[2][2] = best_mutation
print('The best mutation prob is :', best_mutation)
plt.legend()
plt.grid()
plt.title('Mutation prob for Genetic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('ga_mutation_peak.png')

plt.figure(15)
k=[]
mutation_prob = best_mutation
pop_sizes=[200, 400, 600, 800]
for pop_size in pop_sizes:
    ga_fit, ga_curve = ga_plot(problem, mutation_prob, pop_size, max_iter)
    k.append(ga_fit)
    plt.plot(ga_curve[:,0], label='pop: {}'.format(pop_size))
best_pop = pop_sizes[np.argmax(k)]
best_params[2][3] = best_pop
print('The best pop is :', best_pop)
plt.legend()
plt.grid()
plt.title('Pop size for Genetic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('ga_pop_peak.png')

plt.figure(16)
k=[]
pop_size=100
keep_pcts = [0.1, 0.2, 0.3]
for keep_pct in keep_pcts:
    mimic_fit, mimic_curve = mimic_plot(problem, keep_pct, pop_size, max_iter)
    k.append(mimic_fit)
    plt.plot(mimic_curve[:,0], label='keep_pct: {}'.format(keep_pct))
best_keep_pct = keep_pcts[np.argmax(k)]
best_params[2][4] = best_keep_pct
print('The best keep pct is :', best_keep_pct)
plt.legend()
plt.grid()
plt.title('Keep pct for Mimic algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('mimic_keepct_peak.png')
plt.figure(17)
k=[]
keep_pct = best_keep_pct
pop_sizes=[200, 400, 600, 800]
for pop_size in pop_sizes:
    mimic_fit, mimic_curve = mimic_plot(problem, keep_pct, pop_size, max_iter)
    k.append(mimic_fit)
    plt.plot(mimic_curve[:,0], label='pop: {}'.format(pop_size))
best_pop = pop_sizes[np.argmax(k)]
best_params[2][5] = best_pop
print('The best pop is :', best_pop)
plt.legend()
plt.grid()
plt.title('Pop size for Mimic Algorithm')
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.savefig('mimic_pop_peak.png')


# compute the accuracy and time
# Flip-Flop
N=100
fitness = [mlrose_hiive.FlipFlop(), mlrose_hiive.CustomFitness(queens_max), mlrose_hiive.FourPeaks(t_pct=0.1)]
problem = mlrose_hiive.DiscreteOpt(length = N, fitness_fn = fitness[0], maximize = True, max_val = 2)
wall_time = []
best_score=[]
start_time = time.time()
rhc_best_fitness, rhc_fitness_curve = rhc_plot(problem, restart=int(best_params[0][0]), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(rhc_best_fitness)

start_time = time.time()
sa_best_fitness, sa_fitness_curve = sa_plot(problem, schedule=mlrose_hiive.GeomDecay(), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(sa_best_fitness)

start_time = time.time()
ga_best_fitness, ga_fitness_curve = ga_plot(problem, mutation_prob=float(best_params[0][2]), pop_size=int(best_params[0][3]), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(ga_best_fitness)

start_time = time.time()
mimic_best_fitness, mimic_fitness_curve = mimic_plot(problem, keep_pct=float(best_params[0][4]), pop_size=int(best_params[0][5]), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(mimic_best_fitness)

algorithms = ['RHC', 'SA', 'GA', 'Mimic']
x = np.arange(len(algorithms))
plt.figure(18)
bar = plt.bar(x, wall_time)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.2f'%height, ha='center', va='bottom', size=12)
plt.title('Training Times for four algorithms')
plt.xticks(x, algorithms)
plt.ylabel('Training Time (seconds)')
plt.tight_layout()
plt.savefig('flipflop_train_time.png')

plt.figure(19)
bar = plt.bar(x, best_score)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.2f'%height, ha='center', va='bottom', size=12)
plt.title('Fitness for four algorithms')
plt.xticks(x, algorithms)
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('flipflop_train_fitness.png')

plt.figure(20)
plt.plot(rhc_fitness_curve[:,0], label='RHC')
plt.plot(sa_fitness_curve[:,0], label='SA')
plt.plot(ga_fitness_curve[:,0], label='GA')
plt.plot(mimic_fitness_curve[:,0], label='Mimic')
plt.xlabel('iteration')
plt.ylabel('fitness')
plt.title('Fitness Curve')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('flipflop_fitness_curve.png')

print('Train time for each algorithm :', wall_time)
print('Fitness score for each algorithm :', best_score)

# NQueens
N=50
problem = mlrose_hiive.DiscreteOpt(length = N, fitness_fn = fitness[1], maximize = True, max_val = N)
wall_time = []
best_score=[]
start_time = time.time()
rhc_best_fitness, rhc_fitness_curve = rhc_plot(problem, restart=int(best_params[1][0]), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(rhc_best_fitness)

start_time = time.time()
sa_best_fitness, sa_fitness_curve = sa_plot(problem, schedule=mlrose_hiive.GeomDecay(), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(sa_best_fitness)

start_time = time.time()
ga_best_fitness, ga_fitness_curve = ga_plot(problem, mutation_prob=float(best_params[1][2]), pop_size=int(best_params[1][3]), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(ga_best_fitness)

start_time = time.time()
mimic_best_fitness, mimic_fitness_curve = mimic_plot(problem, keep_pct=float(best_params[1][4]), pop_size=int(best_params[1][5]), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(mimic_best_fitness)

algorithms = ['RHC', 'SA', 'GA', 'Mimic']
x = np.arange(len(algorithms))
plt.figure(21)
bar = plt.bar(x, wall_time)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.5f'%height, ha='center', va='bottom', size=12)
plt.title('Training Times for four algorithms')
plt.xticks(x, algorithms)
plt.ylabel('Training Time (seconds)')
plt.tight_layout()
plt.savefig('Queens_train_time.png')

plt.figure(22)
bar = plt.bar(x, best_score)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.2f'%height, ha='center', va='bottom', size=12)
plt.title('Fitness for four algorithms')
plt.xticks(x, algorithms)
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('Queens_train_fitness.png')

plt.figure(23)
plt.plot(rhc_fitness_curve[:,0], label='RHC')
plt.plot(sa_fitness_curve[:,0], label='SA')
plt.plot(ga_fitness_curve[:,0], label='GA')
plt.plot(mimic_fitness_curve[:,0], label='Mimic')
plt.xlabel('iteration')
plt.ylabel('fitness')
plt.title('Fitness Curve')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Queens_fitness_curve.png')

print('Train time for each algorithm :', wall_time)
print('Fitness score for each algorithm :', best_score)

# 4-Peaks
N=100
problem = mlrose_hiive.DiscreteOpt(length = N, fitness_fn = fitness[2], maximize = True, max_val = 2)
wall_time = []
best_score=[]
start_time = time.time()
rhc_best_fitness, rhc_fitness_curve = rhc_plot(problem, restart=int(best_params[2][0]), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(rhc_best_fitness)

start_time = time.time()
sa_best_fitness, sa_fitness_curve = sa_plot(problem, schedule=mlrose_hiive.GeomDecay(), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(sa_best_fitness)

start_time = time.time()
ga_best_fitness, ga_fitness_curve = ga_plot(problem, mutation_prob=float(best_params[2][2]), pop_size=int(best_params[2][3]), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(ga_best_fitness)

start_time = time.time()
mimic_best_fitness, mimic_fitness_curve = mimic_plot(problem, keep_pct=float(best_params[2][4]), pop_size=int(best_params[2][5]), max_iter=1000)
end_time = time.time()
wall_time.append(end_time-start_time)
best_score.append(mimic_best_fitness)

algorithms = ['RHC', 'SA', 'GA', 'Mimic']
x = np.arange(len(algorithms))
plt.figure(24)
bar = plt.bar(x, wall_time)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.5f'%height, ha='center', va='bottom', size=12)
plt.title('Training Times for four algorithms')
plt.xticks(x, algorithms)
plt.ylabel('Training Time (seconds)')
plt.tight_layout()
plt.savefig('Peaks_train_time.png')

plt.figure(25)
bar = plt.bar(x, best_score)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.2f'%height, ha='center', va='bottom', size=12)
plt.title('Fitness for four algorithms')
plt.xticks(x, algorithms)
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('Peaks_train_fitness.png')

plt.figure(26)
plt.plot(rhc_fitness_curve[:,0], label='RHC')
plt.plot(sa_fitness_curve[:,0], label='SA')
plt.plot(ga_fitness_curve[:,0], label='GA')
plt.plot(mimic_fitness_curve[:,0], label='Mimic')
plt.xlabel('iteration')
plt.ylabel('fitness')
plt.title('Fitness Curve')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Peaks_fitness_curve.png')

print('Train time for each algorithm :', wall_time)
print('Fitness score for each algorithm :', best_score)
