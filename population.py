import numpy as np

def parConn_adjust(parConn):
    for i,dimen in enumerate(parConn):
        if dimen == 0:
            if i >= 5:
                parConn[i] = 32
            else:
                parConn[i] = 2**i
    return parConn

def initialize_population(params):
    pop_size = params['pop_size']
    init_max_length = params['init_max_length']
    mean_length = params['mean_length']
    stddev_length = params['stddev_length']
    max_pool = params['max_pool']
    image_channel = params['image_channel']
    max_output_channel = params['max_output_channel']
    population = []
    for _ in range(pop_size):
        num_net = int(np.random.normal(mean_length, stddev_length))
        while num_net > init_max_length:
            num_net = int(np.random.normal(mean_length, stddev_length))
        num_pool = np.random.randint(1, max_pool + 1)
        num_conv = num_net - num_pool
        while num_conv <=0:
            num_pool = np.random.randint(1, max_pool + 1)
            num_conv = num_net - num_pool

        # particle of architecture
        # find the position where the pooling layer can be connected
        availabel_positions = list(range(1,num_net))
        np.random.shuffle(availabel_positions)
        select_positions = np.sort(availabel_positions[0:num_pool]) # the positions of pooling layers in the net
        particle_archit = []
        for i in range(num_net):
            if i in select_positions:
                code_pool = np.random.randint(32, 48)
                particle_archit.append(code_pool)
            else:
                code_conv = np.random.randint(0, max_output_channel)
                particle_archit.append(code_conv)

        # particle of shortcut connection
        # length of the particle is num_net
        particle_conn = []
        for i in range(num_net):
            conn = np.random.randint(1, 2**(i+1))
            if conn >63:
                bin_code = bin(conn)[2:]  # remove 0b head
                new_bin_code = bin_code[-6:]
                conn = int(new_bin_code, 2)
            particle_conn.append(conn)
        particle_conn = parConn_adjust(particle_conn)  # change zeros to non-zeros
        population.append([particle_archit, particle_conn])
    return population

def test_population():
    params = {}
    params['pop_size'] = 20
    params['init_max_length'] = 20
    params['mean_length'] = 15
    params['stddev_length'] = 3
    params['max_pool'] = 2
    params['image_channel'] = 3
    params['max_output_channel'] = 16
    pop = initialize_population(params)
    print(pop)

if __name__ == '__main__':
    test_population()
