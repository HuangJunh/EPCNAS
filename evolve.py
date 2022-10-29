"""
    PSO进化实现。包含对每个粒子的：
    1.particle alignment
    2.velocity calculation
    3.particle updating
    4.architecture evolving
    只做一次迭代
"""
import numpy as np
import copy
import math
def del_parConn_bit(parConn, i_start, i_bit):
    """
    随着particle_archit的长度变化(本函数针对缩短)，需相应对particle_conn做调整。如删除了particle_archit的第3(idx=2)层，
    那么particle_conn的第3(idx=2)维也要删去，同时对于particle_conn(原idx=2维已被删)中现idx=2及其之后的conn code中
    第idx=2+1=3个bit删去。
    :param parConn: 编码connection的粒子 code (已去除被删维之后)
    :param i_start: 从粒子的那一维开始操作，只有被删维度之后的维度才需要del操作。
    :param i_bit: 删去该维度二进制编码的第i的bit。
    :return: 更新后的particle_conn
    """
    for i in range(i_start, len(parConn)):
        dimen_conn = parConn[i]
        bin_code = "".join([str((dimen_conn >> y) & 1) for y in range(i + 1, -1, -1)])
        new_bin_code = bin_code[:i_bit] + bin_code[i_bit + 1:]
        parConn[i] = int(new_bin_code, 2)
    return parConn

def del_velConn_bit(verConn, i_start, i_bit):
    """
    跟del_parConn_bit一样，删除了particle一层后，就要删除其velocity_archi,parConn和verConn的对应位置，然后对parConn和verConn中
    编码的一些bit位也要删去。
    :param verConn: particle_conn的velocity [[v00], [v10, v11], [v20, v21, v22], ... [vn0, vn1, ... vnn]]
    :param i_start: 开始删除的位置
    :param i_bit: 每一个维度上要删除的bit的位置
    :return:
    """
    for i in range(i_start, len(verConn)):
        del verConn[i][i_bit]
    return verConn

def insert_parConn_bit(parConn, i_start, i_bit):
    """
    随着particle_archit的长度变化(本函数针对增长)，需相应对particle_conn做调整。如增加了particle_archit的第3（idx=2）层，
    那么particle_conn也要增加第3（idx=2）维，同时对于particle_conn(已增加idx=2维)中现idx+1=3及其之后的conn code中第idx=2需加1bit 0
    :param parConn: 编码shortcut connection的粒子
    :param i_start: 从粒子的那一维开始操作，只有被增加维度之后的维度才需要add操作。
    :param i_bit: 删去该维度二进制编码的第i的bit。
    :return: 更新后的particle_conn
    """
    for i in range(i_start, len(parConn)):
        dimen_conn = parConn[i]
        bin_code = "".join([str((dimen_conn >> y) & 1) for y in range(i, -1, -1)])
        new_bin_code = bin_code[:i_bit]+'0'+ bin_code[i_bit:]
        parConn[i] = int(new_bin_code, 2)
    return parConn

def insert_velConn_bit(velConn, i_start, i_bit):
    """
    跟insert_parConn_bit一样，增加了particle一层后，就要增加其velocity_archi,parConn和verConn的对应位置，然后对parConn和verConn中
    编码的一些bit位也要增加。
    :param velConn:
    :param i_start:
    :param i_bit:
    :return:
    """
    for i in range(i_start, len(velConn)):
        velConn[i].insert(i_bit, 0)
    return velConn

def parConn_adjust(parConn):
    """
    在pso结束之后（apso或cpso都需要），可能出现conn的particle为0的情况，如果出现这种情况，就将其值改为16（就是对应连接到其往前5位的node上）
    注意，经过这么处理，第1个evolved node肯定会且只会连接到input node的
    :param parConn:
    :return:
    """
    for i,dimen in enumerate(parConn):
        if dimen == 0:
            if i >= 5:
                parConn[i] = 32
            else:
                parConn[i] = 2**i
    return parConn

def apso(particle, gbest, pbest, velocity, params):
    """
    pso for architecture evolution
    flexible variable-length PSO，经过particle alignment，velocity calculation，然后获得一个particle之后，还需要对particle中
    的invalid做调整，即根据制定的规则，赋予这些数值意义，从而调整particle长度，同时也需要对velocity对应位置进行调整。在evolve particle_archit
    的同时相应调整particle_conn,以保证后面cPSO能顺利进行
    :param cur_particle: 当前粒子,含结构与连接两部分
    :param gbest: 当前全局最优粒子
    :param pbest: 当前粒子的历史最优粒子
    :param velocity: 上一代velocity,含结构与连接两部分
    :param params: CNN配置限制参数，尤其是约束pool num。
    :return: particle_archit, particle_conn, velocity_archit, velocity_conn
    """
    cur_particle, particle_conn = particle
    cur_velocity, velocity_conn = velocity
    # cur_velocity = [v0, v1, ..., vn]
    # velocity_conn = [[v00], [v10, v11], [v20, v21, v22], ... [vn0, vn1, ... vnn]]

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
        # new_pbest = list(map(int, new_pbest))

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
    new_particle = list(map(int, cur_particle + new_velocity))  #particle里面的数必须为整数
    new_velocity = list(new_velocity)

    # 4.architecture evolving
    # 调整particle值(invalid值的处理)，生成新individual
    # 首层必须为conv（不能为pool或者是value<0(会被去除)）,当粒子长度大于1都会执行去除操作
    while len(new_particle) > 1 and (new_particle[0] < 0 or new_particle[0] > 31):
        del new_particle[0]
        del new_velocity[0]

        # adjust particle_conn
        del particle_conn[0]
        del velocity_conn[0]
        particle_conn = del_parConn_bit(particle_conn, 0, 1)
        velocity_conn = del_velConn_bit(velocity_conn, 0, 1)

    # 如果上面操作导致得到的粒子长度只剩1，且还存在首层为pool或者value<0的情况，则将该长度为1的粒子置为1（类似放弃该粒子），避免程序运行出错
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
            if not j == len(particle_conn):  # 如果删去的层是最后一层，则不需要再调整连接编码的bit
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
            particle_conn.insert(j+1, 1)    # 插入的dimen会与前一dimen相连，所以code是1
            velocity_conn.insert(j+1, [0]*(j+2))
            if not j == len(particle_conn)-2:   # 如果增加的层是最后一层，则不需要再调整连接编码的bit
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
            if not j == len(particle_conn) - 2:  # 如果增加的层是最后一层，则不需要再调整连接编码的bit
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
            if not j == len(particle_conn) - 2:  # 如果增加的层是最后一层，则不需要再调整连接编码的bit
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
    selected_idx = np.random.choice(pool_idx, 1, replace=False) #每次就删一个
    del new_particle[selected_idx[0]]
    del new_velocity[selected_idx[0]]
    del particle_conn[selected_idx[0]]
    del velocity_conn[selected_idx[0]]
    if not selected_idx[0] == len(particle_conn):  # 如果删去的层是最后一层，则不需要再调整连接编码的bit
        particle_conn = del_parConn_bit(particle_conn, selected_idx[0], selected_idx[0]+1)
        velocity_conn = del_velConn_bit(velocity_conn, selected_idx[0], selected_idx[0]+1)
    return new_particle, new_velocity, particle_conn, velocity_conn

def fsigmoid(x):
    return 1/(1+np.exp(-x))

def cpso(cur_particle, gbest, pbest, velocity, params, offsets):
    """
    pso for shortcut connection evolution,只evolve particle_conn即可，粒子长度固定不变
    :param cur_particle: 当前粒子particle_conn,只包含shortcut connection representation
    :param gbest: 当前全局最优粒子的connection编码
    :param pbest: 当前粒子的历史最优粒子的connection编码
    :param velocity: 上一代velocity_conn
    :param params: CNN配置限制参数，尤其是约束pool num。
    :return: new_particle, new_velocity, 其中new velocity的维度要跟new_particle一致
    """
    cur_len = len(cur_particle)
    pbest_len = len(pbest)
    gbest_len = len(gbest)

    # 1.particle alignment
    # offset1 = np.random.randint(0, abs(cur_len - pbest_len) + 1)
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
        # new_pbest = list(map(int, new_pbest))

    # offset2 = np.random.randint(0, abs(cur_len - gbest_len) + 1)
    if gbest_len >= cur_len:
        new_gbest = np.asarray(gbest[offset2:offset2 + cur_len])
    else:
        new_gbest = np.zeros(cur_len)
        new_gbest[offset2:offset2 + gbest_len] = gbest
        # new_gbest = list(map(int, new_gbest))

    # 2.velocity calculation
    # velocity_conn = [[v00], [v10, v11], [v20, v21, v22], ... [vn0, vn1, ... vnn]]

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
    # particle长度是不变的，但会对里面的值做处理。如果值小于0，将将其置为0.其它值则是取其后面对应位数的bit出来，作为新值。
    # 如idx=0对应bit数是1，只取该数的二进制值的最后一位，然后再转换为十进制。
    for i in range(len(new_particle)):
        if new_particle[i] < 0:
            new_particle[i] = 0
        else:
            # bin_code = "".join([str((new_particle[i] >> y) & 1) for y in range(i, -1, -1)])
            bin_code = bin(new_particle[i])[2:] #去掉0b开头
            new_bin_code = bin_code[-(i+1):]
            new_particle[i] = int(new_bin_code, 2)
            
            if new_particle[i] > 63:  # 有值超过7bit大小
                bin_code = bin(new_particle[i])[2:]  # 去掉0b开头
                new_bin_code = bin_code[-6:]
                new_particle[i] = int(new_bin_code, 2)
    new_particle = parConn_adjust(new_particle) # 将其中的0值调整为非0
    return new_particle, new_velocity
