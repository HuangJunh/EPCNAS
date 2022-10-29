# !/usr/bin/python
# -*- coding: utf-8 -*-

from utils import Utils, Log
from utils_32 import Utils as Utils_32
from utils_64 import Utils as Utils_64
from population import initialize_population
from evaluate import decode, fitnessEvaluate
from evolve import apso,cpso
import copy, os, time
factors = [1, 2, 4]
batch_size_set=[64,64,64]
weight_decay_set=[4e-5,4e-5,4e-5]
epochs_for_apso = range(0,101)
epochs_for_cpso = range(0,101)

def create_directory():
    dirs = ['./log', './populations', './scripts']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def fitness_evaluate(population, curr_gen):
    filenames = []
    for i, particle in enumerate(population):
        filename = decode(particle, curr_gen, i)
        filenames.append(filename)

    acc_set = fitnessEvaluate(filenames, curr_gen, is_test=False)
    return acc_set

def evolve(population, gbest_individual, pbest_individuals, velocity_set, params):
    offspring = []
    new_velocity_set = []
    gen_no = params['gen_no']
    for i,particle in enumerate(population):
        particle_archit, particle_conn = particle
        velocity_archit, velocity_conn = velocity_set[i]
        is_apso = False
        if gen_no in epochs_for_apso:
            particle_archit, particle_conn, velocity_archit, velocity_conn, offsets = apso(particle, gbest_individual[0], pbest_individuals[i][0], velocity_set[i], params)
            is_apso = True
        if gen_no in epochs_for_cpso:
            if is_apso:
                particle_conn, velocity_conn = cpso(particle_conn, gbest_individual[1], pbest_individuals[i][1], velocity_conn, params, offsets)
            else:
                particle_conn, velocity_conn = cpso(particle_conn, gbest_individual[1], pbest_individuals[i][1], velocity_conn, params, None)
        offspring.append([particle_archit, particle_conn])
        new_velocity_set.append([velocity_archit, velocity_conn])
    return offspring, new_velocity_set

def update_best_particle(population, acc_set, gbest, pbest):
    if not pbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_accSet = copy.deepcopy(acc_set)
        gbest_individual, gbest_acc = getGbest([pbest_individuals, pbest_accSet])
    else:
        gbest_individual, gbest_acc = gbest
        pbest_individuals, pbest_accSet = pbest
        for i,acc in enumerate(acc_set):
            if acc > pbest_accSet[i]:
                pbest_individuals[i] = copy.deepcopy(population[i])
                pbest_accSet[i] = copy.deepcopy(acc)
            if acc > gbest_acc:
                gbest_individual = copy.deepcopy(population[i])
                gbest_acc = copy.deepcopy(acc)

    return [gbest_individual, gbest_acc], [pbest_individuals, pbest_accSet]

def getGbest(pbest):
    pbest_individuals, pbest_accSet = pbest
    gbest_acc = 0
    gbest = None
    for i,indi in enumerate(pbest_individuals):
        if pbest_accSet[i] > gbest_acc:
            gbest = copy.deepcopy(indi)
            gbest_acc = copy.deepcopy(pbest_accSet[i])
    return gbest, gbest_acc

def scale_individual(gbest_individual, factor):
    particle_a, _ = gbest_individual
    final_individual = copy.deepcopy(gbest_individual)
    for i in range(len(particle_a)):
        final_individual[0][i] = (gbest_individual[0][i]+1)*factor-1
    return final_individual

def fitness_test(final_individuals, factors):
    filenames = []
    for i,final_individual in enumerate(final_individuals):
        if factors[i] == 1:
            filename = Utils.generate_pytorch_file(final_individual, -1, i)
        elif factors[i] == 2:
            filename = Utils_32.generate_pytorch_file(final_individual, -1, i)
        elif factors[i] == 4:
            filename = Utils_64.generate_pytorch_file(final_individual, -1, i)
        filenames.append(filename)
    acc_set = fitnessEvaluate(filenames, -1, True, batch_size_set, weight_decay_set)
    return acc_set

def evolveCNN(params):
    gen_no = 0
    Log.info('Initialize...')
    start = time.time()
    population = initialize_population(params)

    Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
    acc_set = fitness_evaluate(population, gen_no)
    Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

    [gbest_individual, gbest_acc], [pbest_individuals, pbest_accSet] = update_best_particle(population, acc_set, gbest=None, pbest=None)

    Log.info('EVOLVE[%d-gen]-Finish the updating' % (gen_no))

    Utils.save_population_and_acc('population', population, acc_set, gen_no)
    Utils.save_population_and_acc('pbest', pbest_individuals, pbest_accSet, gen_no)
    Utils.save_population_and_acc('gbest', [gbest_individual], [gbest_acc], gen_no)

    gen_no += 1
    velocity_set = []  # [[velocity_a, velocity_c], [velocity_a, velocity_c],...]
    for ii in range(len(population)):
        velocity1 = []
        velocity2 = []
        for jj in range(len(population[ii][0])):
            velocity1.append(1)
            velocity2.append([0] * (jj + 1))
        velocity_set.append([velocity1, velocity2])

    for curr_gen in range(gen_no, params['num_iteration']):
        params['gen_no'] = curr_gen

        Log.info('EVOLVE[%d-gen]-Begin pso evolution' % (curr_gen))
        population, velocity_set = evolve(population, gbest_individual, pbest_individuals, velocity_set, params)
        Log.info('EVOLVE[%d-gen]-Finish pso evolution' % (curr_gen))

        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (curr_gen))
        acc_set = fitness_evaluate(population, curr_gen)
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (curr_gen))

        [gbest_individual, gbest_acc], [pbest_individuals, pbest_accSet] = update_best_particle(population, acc_set, gbest=[gbest_individual, gbest_acc], pbest=[pbest_individuals, pbest_accSet])
        Log.info('EVOLVE[%d-gen]-Finish the updating' % (curr_gen))

        Utils.save_population_and_acc('population', population, acc_set, curr_gen)
        Utils.save_population_and_acc('pbest', pbest_individuals, pbest_accSet, curr_gen)
        Utils.save_population_and_acc('gbest', [gbest_individual], [gbest_acc], curr_gen)

    end = time.time()
    Log.info('Total Search Time: %.2f seconds' % (end-start))
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    Log.info("%02dh:%02dm:%02ds" % (h, m, s))

    # final training and test on testset
    final_individuals = []
    for fac in factors:
        final_individual = scale_individual(gbest_individual,fac)
        final_individuals.append(final_individual)
    gbest_acc_set = fitness_test(final_individuals, factors)
    for i, final_individual in enumerate(final_individuals):
        Log.info('Accuracy=[%.5f]'%(gbest_acc_set[i]))
        if factors[i] == 1:
            Utils.save_population_and_acc('final_gbest_16', [final_individual], [gbest_acc_set[i]], -1)
        elif factors[i] == 2:
            Utils_32.save_population_and_acc('final_gbest_32', [final_individual], [gbest_acc_set[i]], -1)
        elif factors[i] == 4:
            Utils_64.save_population_and_acc('final_gbest_64', [final_individual], [gbest_acc_set[i]], -1)

if __name__ == '__main__':
    create_directory()
    params = Utils.get_init_params()
    evoCNN = evolveCNN(params)

