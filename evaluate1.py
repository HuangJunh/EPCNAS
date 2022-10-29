# !/usr/bin/python
# -*- coding: utf-8 -*-

from utils import Utils, Log, GPUTools
from multiprocessing import Process
import importlib
import sys,os, time
import numpy as np
from asyncio.tasks import sleep

def decode(particle, curr_gen, id):

    pytorch_filename = Utils.generate_pytorch_file(particle, curr_gen, id)

    return pytorch_filename


def fitnessEvaluate(filenames, curr_gen, is_test, batch_size_set=None, weight_decay_set=None):
    acc_set = np.zeros(len(filenames))
    has_evaluated_offspring = False
    for i,file_name in enumerate(filenames):
        has_evaluated_offspring = True
        time.sleep(20)
        gpu_id = GPUTools.detect_available_gpu_id()
        while gpu_id is None:
            time.sleep(20)
            gpu_id = GPUTools.detect_available_gpu_id()
        if gpu_id is not None:
            Log.info('Begin to train %s' % (file_name))
            module_name = 'scripts.%s' % (file_name)
            if module_name in sys.modules.keys():
                Log.info('Module:%s has been loaded, delete it' % (module_name))
                del sys.modules[module_name]
                _module = importlib.import_module('.', module_name)
            else:
                _module = importlib.import_module('.', module_name)
            _class = getattr(_module, 'RunModel')
            cls_obj = _class()
            if batch_size_set:
                p = Process(target=cls_obj.do_work, args=('%d' % (gpu_id), curr_gen, file_name, is_test, batch_size_set[i], weight_decay_set[i]))
            else:
                p = Process(target=cls_obj.do_work, args=('%d' % (gpu_id), curr_gen, file_name, is_test))
            p.start()

    if has_evaluated_offspring:
        all_finished = False
        times = 0
        while all_finished is not True:
            time.sleep(60)
            all_finished = GPUTools.all_gpu_available()
            times += 1

    if has_evaluated_offspring:
        file_name = './populations/after_%02d.txt' % (curr_gen)
        assert os.path.exists(file_name) == True
        f = open(file_name, 'r')
        fitness_map = {}
        for line in f:
            if len(line.strip()) > 0:
                line = line.strip().split('=')
                fitness_map[line[0]] = float(line[1])
        f.close()

        for i in range(len(acc_set)):
            if filenames[i] not in fitness_map:
                Log.warn(
                    'The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 120 seconds' % (
                        filenames[i], file_name))
                sleep(120)
            acc_set[i] = fitness_map[filenames[i]]

    else:
        Log.info('None offspring has been evaluated')

    return list(acc_set)


