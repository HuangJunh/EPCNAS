import numpy as np
import copy
import math
def del_parConn_bit(parConn, i_start, i_bit):
    for i in range(i_start, len(parConn)):
        dimen_conn = parConn[i]
        bin_code = "".join([str((dimen_conn >> y) & 1) for y in range(i + 1, -1, -1)])
        new_bin_code = bin_code[:i_bit] + bin_code[i_bit + 1:]
        parConn[i] = int(new_bin_code, 2)
    return parConn

def del_velConn_bit(verConn, i_start, i_bit):
    for i in range(i_start, len(verConn)):
        del verConn[i][i_bit]
    return verConn

def insert_parConn_bit(parConn, i_start, i_bit):
    for i in range(i_start, len(parConn)):
        dimen_conn = parConn[i]
        bin_code = "".join([str((dimen_conn >> y) & 1) for y in range(i, -1, -1)])
        new_bin_code = bin_code[:i_bit]+'0'+ bin_code[i_bit:]
        parConn[i] = int(new_bin_code, 2)
    return parConn

def insert_velConn_bit(velConn, i_start, i_bit):
    for i in range(i_start, len(velConn)):
        velConn[i].insert(i_bit, 0)
    return velConn

def parConn_adjust(parConn):
    for i,dimen in enumerate(parConn):
        if dimen == 0:
            if i >= 5:
                parConn[i] = 32
            else:
                parConn[i] = 2**i
    return parConn

def apso(particle, gbest, pbest, velocity, params):
    cur_particle, particle_conn = particle
    cur_velocity, velocity_conn = velocity

    cur_len = len(cur_particle)
    pbest_len = len(pbest)
    gbest_len = len(gbest)

    # 1.particle alignment
    offset1 = np.random.randint(0, abs(cur_len - pbest_len) + 1)
    if pbest_len >= cur_len:
        new_pbest = np.asarray(pbest[offset1:offset1 + cur_len])
    else:
        new_pbest = np.zeros(cur_len)
        new_pbest[offset1:offset1 + pbest_len] = pbest

    offset2 = np.random.randint(0, abs(cur_len - gbest_len) + 1)
    if gbest_len >= cur_len:
        new_gbest = np.asarray(gbest[offset2:offset2 + cur_len])
    else:
        new_gbest = np.zeros(cur_len)
        new_gbest[offset2:offset2 + gbest_len] = gbest
        # new_gbest = list(map(int, new_gbest))

    # 2.velocity calculation
    w, c1, c2 = 0.7298, 1.49618, 1.49618
    r1 = np.random.random(cur_len)
    r2 = np.random.random(cur_len)
    new_velocity = np.asarray(cur_velocity) * w + c1 * r1 * (new_pbest - cur_particle) + c2 * r2 * (new_gbest - cur_particle)

    # 3.particle updating
    new_particle = list(map(int, cur_particle + new_velocity))
    new_velocity = list(new_velocity)

    # 4.architecture evolving
    while len(new_particle) > 1 and (new_particle[0] < 0 or new_particle[0] > 31):
        del new_particle[0]
        del new_velocity[0]

        # adjust particle_conn
        del particle_conn[0]
        del velocity_conn[0]
        particle_conn = del_parConn_bit(particle_conn, 0, 1)
        velocity_conn = del_velConn_bit(velocity_conn, 0, 1)

    if new_particle[0] < 0 or new_particle[0] > 31:
        new_particle[0] = 3

    j = 0
    while j < len(new_particle):
        if new_particle[j] < 0:
            del new_particle[j]
            del new_velocity[j]
            # adjust particle_conn
            del particle_conn[j]
            del velocity_conn[j]
            if not j == len(particle_conn):
                particle_conn = del_parConn_bit(particle_conn, j, j+1)
                velocity_conn = del_velConn_bit(velocity_conn, j, j+1)
            j -= 1
        elif 16 <= new_particle[j] <= 31:
            value1 = new_particle[j]
            new_particle[j] = value1 // 2
            new_particle.insert(j + 1, value1 - value1 // 2)

            value2 = new_velocity[j]
            new_velocity[j] = value2 // 2
            new_velocity.insert(j + 1, value2 - value2 // 2)

            # adjust particle_conn
            particle_conn.insert(j+1, 1)
            velocity_conn.insert(j+1, [0]*(j+2))
            if not j == len(particle_conn)-2:
                particle_conn = insert_parConn_bit(particle_conn, j+2, j+1)
                velocity_conn = insert_velConn_bit(velocity_conn, j+2, j+1)
        elif 48 <= new_particle[j] <= 63:
            # if dimen falls into this range, we will have two different pooling types, avg+max
            value1 = new_particle[j]
            gap = value1 - 47
            new_particle[j] = 39 + gap - gap//2   # avg pooling
            new_particle.insert(j + 1, 40 - gap//2)    # max pooling

            value2 = new_velocity[j]
            new_velocity[j] = value2 - value2 // 2
            new_velocity.insert(j + 1, value2 // 2)

            # adjust particle_conn
            particle_conn.insert(j+1, 1)
            velocity_conn.insert(j+1, [0]*(j+2))
            if not j == len(particle_conn) - 2:
                particle_conn = insert_parConn_bit(particle_conn, j + 2, j + 1)
                velocity_conn = insert_velConn_bit(velocity_conn, j + 2, j + 1)
        while new_particle[j] > 63:
            value1 = new_particle[j]
            new_particle[j] = value1 // 2
            new_particle.insert(j + 1, value1 - value1 // 2)

            value2 = new_velocity[j]
            new_velocity[j] = value2 // 2
            new_velocity.insert(j + 1, value2 - value2 // 2)

            # adjust particle_conn
            particle_conn.insert(j+1, 1)
            velocity_conn.insert(j+1, [0]*(j+2))
            if not j == len(particle_conn) - 2:
                particle_conn = insert_parConn_bit(particle_conn, j + 2, j + 1)
                velocity_conn = insert_velConn_bit(velocity_conn, j + 2, j + 1)
            j-=1
        j += 1

    pool_num = _calculate_pool_numbers(new_particle)
    while pool_num > params['max_pool']:
        new_particle, new_velocity, particle_conn, velocity_conn = cut_pool(new_particle, new_velocity, particle_conn, velocity_conn)
        pool_num = _calculate_pool_numbers(new_particle)
    particle_conn = parConn_adjust(particle_conn)
    return new_particle, particle_conn, new_velocity, velocity_conn, [offset1,offset2]


def _calculate_pool_numbers(particle):
    num_pool = 0
    for dimension in particle:
        if 32<=dimension<=47:
            num_pool+=1
    return num_pool

def cut_pool(new_particle, new_velocity, particle_conn, velocity_conn):
    pool_idx = [list(enumerate(new_particle))[i][0] for i in range(len(new_particle)) if 32 <= new_particle[i] <= 47]
    selected_idx = np.random.choice(pool_idx, 1, replace=False)
    del new_particle[selected_idx[0]]
    del new_velocity[selected_idx[0]]
    del particle_conn[selected_idx[0]]
    del velocity_conn[selected_idx[0]]
    if not selected_idx[0] == len(particle_conn):
        particle_conn = del_parConn_bit(particle_conn, selected_idx[0], selected_idx[0]+1)
        velocity_conn = del_velConn_bit(velocity_conn, selected_idx[0], selected_idx[0]+1)
    return new_particle, new_velocity, particle_conn, velocity_conn

def fsigmoid(x):
    return 1/(1+np.exp(-x))

def cpso(cur_particle, gbest, pbest, velocity, params, offsets):
    cur_len = len(cur_particle)
    pbest_len = len(pbest)
    gbest_len = len(gbest)

    # 1.particle alignment
    if offsets:
        offset1, offset2 = offsets
        if offset1>abs(cur_len - pbest_len):
            offset1 = abs(cur_len - pbest_len)
        if offset2>abs(cur_len - gbest_len):
            offset2 = abs(cur_len - gbest_len)
    else:
        offset1, offset2 = 0, 0

    if pbest_len >= cur_len:
        new_pbest = np.asarray(pbest[offset1:offset1 + cur_len])
    else:
        new_pbest = np.zeros(cur_len)
        new_pbest[offset1:offset1 + pbest_len] = pbest

    # offset2 = np.random.randint(0, abs(cur_len - gbest_len) + 1)
    if gbest_len >= cur_len:
        new_gbest = np.asarray(gbest[offset2:offset2 + cur_len])
    else:
        new_gbest = np.zeros(cur_len)
        new_gbest[offset2:offset2 + gbest_len] = gbest

    w, c1, c2 = 0.7298, 1.49618, 1.49618
    new_velocity = copy.deepcopy(velocity)
    new_particle = copy.deepcopy(cur_particle)
    for i in range(cur_len):
        cur_particle_bin = "".join([str((int(cur_particle[i]) >> y) & 1) for y in range(i, -1, -1)])
        cur_particle_binList = np.asarray([int(bi) for bi in cur_particle_bin])
        new_pbest_bin = "".join([str((int(new_pbest[i]) >> y) & 1) for y in range(i, -1, -1)])
        new_pbest_binList = np.asarray([int(bi) for bi in new_pbest_bin])
        new_gbest_bin = "".join([str((int(new_gbest[i]) >> y) & 1) for y in range(i, -1, -1)])
        new_gbest_binList = np.asarray([int(bi) for bi in new_gbest_bin])

        r1 = np.random.random(i+1)
        r2 = np.random.random(i+1)
        r3 = np.random.random(i + 1)
        new_velocity[i] = np.asarray(velocity[i]) * w + c1 * r1 * (new_pbest_binList - cur_particle_binList) + c2 * r2 * (new_gbest_binList - cur_particle_binList)

        s_new_velocity = fsigmoid(new_velocity[i])
        new_velocity_prob = r3-s_new_velocity
        new_particle_binList = [1-math.ceil(prob) for prob in new_velocity_prob]
        new_particle_bin = "".join([str(bi) for bi in new_particle_binList])
        new_particle[i] = int(new_particle_bin, 2)

        new_velocity[i] = list(new_velocity[i])

    # 3.particle updating
    new_velocity = list(new_velocity)

    # 4.connection evolving
    for i in range(len(new_particle)):
        if new_particle[i] < 0:
            new_particle[i] = 0
        else:
            # bin_code = "".join([str((new_particle[i] >> y) & 1) for y in range(i, -1, -1)])
            bin_code = bin(new_particle[i])[2:] #remove 0b
            new_bin_code = bin_code[-(i+1):]
            new_particle[i] = int(new_bin_code, 2)
            
            if new_particle[i] > 63:
                bin_code = bin(new_particle[i])[2:]  #remove 0b
                new_bin_code = bin_code[-6:]
                new_particle[i] = int(new_bin_code, 2)
    new_particle = parConn_adjust(new_particle)
    return new_particle, new_velocity
