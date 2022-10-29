# !/usr/bin/python
# -*- coding: utf-8 -*-
import configparser
import logging
import time
import sys
import os
import math
from subprocess import Popen, PIPE
expansion_rates = [3, 4, 6, 6, 4]
kernel_sizes = [3, 5, 5, 7, 7]
class Utils(object):
    @classmethod
    def get_init_params(cls):
        params = {}
        params['pop_size'] = cls.get_params('settings', 'pop_size')
        params['num_iteration'] = cls.get_params('settings', 'num_iteration')
        params['init_max_length'] = cls.get_params('network', 'init_max_length')
        params['mean_length'] = cls.get_params('network', 'mean_length')
        params['stddev_length'] = cls.get_params('network', 'stddev_length')
        params['max_pool'] = cls.get_params('network', 'max_pool')
        params['image_channel'] = cls.get_params('network', 'image_channel')
        params['max_output_channel'] = cls.get_params('network', 'max_output_channel')
        params['num_class'] = cls.get_params('network', 'num_class')
        params['min_epoch_eval'] = cls.get_params('network', 'min_epoch_eval')
        params['epoch_test'] = cls.get_params('network', 'epoch_test')
        return params

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)

    @classmethod
    def get_params(cls, domain, key):
        rs = cls.__read_ini_file(domain, key)
        return int(rs)


    @classmethod
    def save_population_and_acc(cls, type, population, acc_set, gen_no):
        file_name = './populations/' + type + '_%02d.txt' % (gen_no)
        _str = cls.popAndAcc2str(population, acc_set)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()

    @classmethod
    def popAndAcc2str(cls, population, acc_set):
        pop_str = []
        for id, particle in enumerate(population):
            particle_a,particle_c = particle
            inchannels, outchannels, channels_end = cls.calc_in_out_channels(particle_a, particle_c)
            _str = []
            _str.append('indi:%02d' % (id))
            _str.append('particle_a:%s' % (','.join(list(map(str, particle_a)))))
            _str.append('particle_c:%s' % (','.join(list(map(str, particle_c)))))
            particle_c_bins = []
            for i in range(len(particle_c)):
                particle_c_bins.append("".join([str((int(particle_c[i]) >> y) & 1) for y in range(i, -1, -1)]))
            _str.append('particle_c (bin):%s' % ('.'.join(list(map(str, particle_c_bins)))))

            _str.append('eval_acc:%.4f' % (acc_set[id]))
            for number, dimen in enumerate(particle_a):
                in_channel = inchannels[number]
                out_channel = outchannels[number]
                _sub_str = []
                if 0 <= dimen <= 31:
                    _sub_str.append('conv%d'%(number))
                    _sub_str.append('in:%d' % (in_channel))
                    _sub_str.append('out:%d' % (out_channel))

                elif 64 <= dimen <= 95:
                    _sub_str.append('pool%d' % (number))
                    if 64 <= dimen <= 79:
                        _sub_str.append('type:%s' % ('max'))
                    else:
                        _sub_str.append('type:%s' % ('average'))

                input_layers = ''
                bin_code = "".join([str((particle_c[number] >> y) & 1) for y in range(number, -1, -1)])
                for j, bit in enumerate(bin_code):
                    if int(bit) == 1:
                        if j == 0:
                            input_layers += 'conv_begin, '
                        elif 0 <= particle_a[j - 1] <= 31:
                            input_layers += 'conv%d, ' % (j - 1)
                        else:
                            input_layers += 'pool%d, ' % (j - 1)
                _sub_str.append('input:' + input_layers)
                _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))

            # add final concatenate information
            _concat_str = []
            _concat_str.append('final channels:' + str(channels_end) + ', final concatenate layers: ')
            if 0 <= particle_a[-1] <= 31:
                input_layers = 'conv%d, ' % (len(particle_a) - 1)
            else:
                input_layers = 'pool%d, ' % (len(particle_a) - 1)

            layer_indices = cls.obtain_final_output_layers(particle_c)
            for j in layer_indices:
                if j == -1:
                    input_layers += 'conv_begin, '
                elif 0 <= particle_a[j] <= 31:
                    input_layers += 'conv%d, ' % (j)
                else:
                    input_layers += 'pool%d, ' % (j)

            _concat_str.append(input_layers)
            _str.append('%s%s%s' % ('[', ','.join(_concat_str), ']'))

            particle_str = '\n'.join(_str)
            pop_str.append(particle_str)
            pop_str.append('-' * 100)
        return '\n'.join(pop_str)

    @classmethod
    def obtain_final_output_layers(cls, particle_c):
        particle_c_bins = []
        num_output = [0]*(len(particle_c))
        for i in range(len(particle_c)):
            particle_c_bin = "".join([str((int(particle_c[i]) >> y) & 1) for y in range(i, -1, -1)])
            particle_c_bins.append(particle_c_bin)
            for j in range(i+1):
                if particle_c_bin[j] == '1':
                    num_output[j]+=1
        target_output_indices = [list(enumerate(num_output))[i][0]-1 for i in range(len(num_output)) if num_output[i] == 0]

        return target_output_indices

    @classmethod
    def calc_in_out_channels(cls, particle_a, particle_c):
        inchannels = [particle_a[0] + 1]
        outchannels = [particle_a[0] + 1]
        for i in range(1, len(particle_c)):
            val = particle_c[i]
            bin_code = "".join([str((val >> y) & 1) for y in range(i, -1, -1)])
            # print(bin_code)
            # inchannel = outchannels[-1]
            inchannel = 0
            for j, bit in enumerate(bin_code):
                if int(bit) == 1:
                    if j == 0:
                        inchannel += inchannels[0]
                    else:
                        inchannel += outchannels[j - 1]
            inchannels.append(inchannel)

            if 0 <= particle_a[i] <= 31:
                outchannels.append(particle_a[i] + 1)
            else:
                outchannels.append(inchannels[-1])

        layer_indices = cls.obtain_final_output_layers(particle_c)
        channels_end = outchannels[-1]
        for idx in layer_indices:
            if idx == -1:
                channels_end += inchannels[0]
            else:
                channels_end += outchannels[idx]
        return inchannels, outchannels, channels_end

    @classmethod
    def obtain_output_size(cls, particle_a, particle_c):
        output_feature_sizes = [32]
        for i in range(len(particle_a)):
            bin_code = "".join([str((particle_c[i] >> y) & 1) for y in range(i, -1, -1)])
            input_feature_sizes = []
            for j, bit in enumerate(bin_code):
                if int(bit) == 1:
                    input_feature_sizes.append(output_feature_sizes[j])
            input_feature_size = min(input_feature_sizes)
            if 0 <= particle_a[i] <= 31:
                output_feature_sizes.append(input_feature_size)
            else:
                output_feature_size = int((input_feature_size - 1) / 2) + 1
                output_feature_sizes.append(output_feature_size)
        return output_feature_sizes

    @classmethod
    def generate_concat_code(cls, particle_a, particle_c, i, bin_code, forward_list):
        output_feature_sizes = cls.obtain_output_size(particle_a, particle_c)

        curr_input_feature_sizes = [[j - 1, output_feature_sizes[j]] for j, bit in enumerate(bin_code) if int(bit) == 1]
        target_input_feature_size = min([k for _, k in curr_input_feature_sizes])
        for k, [pos, input_feature_size] in enumerate(curr_input_feature_sizes):
            if input_feature_size == target_input_feature_size:
                if pos == -1:
                    forward_list.append('input_%d = out_begin' % (i))
                else:
                    forward_list.append('input_%d = out_%d' % (i, pos))
                del curr_input_feature_sizes[k]
                break

        for pos, input_feature_size in curr_input_feature_sizes:
            if input_feature_size == target_input_feature_size:
                if pos == -1:
                    tar = 'out_begin'
                    _str = 'input_%d = torch.cat((input_%d, %s),1)' % (i, i, tar)
                else:
                    tar = 'out_%d' % (pos)
                    _str = 'input_%d = torch.cat((input_%d, %s),1)' % (i, i, tar)

            else:
                if pos == -1:
                    tar = 'self.sdconv_begin_%d(out_begin)' % (i)
                    _str = 'input_%d = torch.cat((input_%d, %s),1)' % (i, i, tar)
                else:
                    tar = 'self.sdconv_%d_%d(out_%d)' % (pos, i, pos)
                    _str = 'input_%d = torch.cat((input_%d, %s),1)' % (i, i, tar)
            forward_list.append(_str)
        return forward_list

    @classmethod
    def generate_concat_code_outputNode(cls, particle_a, particle_c, i, forward_list):
        output_feature_sizes = cls.obtain_output_size(particle_a, particle_c)
        layer_indices = cls.obtain_final_output_layers(particle_c)
        layer_indices.append(len(particle_c) - 1)
        if len(layer_indices) == 1:
            forward_list.append('input_%d = out_%d' % (len(particle_a), len(particle_a) - 1))
        else:
            curr_input_feature_sizes = [[indx, output_feature_sizes[indx + 1]] for indx in layer_indices]
            target_input_feature_size = min([k for _, k in curr_input_feature_sizes])
            for k, [pos, input_feature_size] in enumerate(curr_input_feature_sizes):
                if input_feature_size == target_input_feature_size:
                    if pos == -1:
                        forward_list.append('input_%d = out_begin' % (i))
                    else:
                        forward_list.append('input_%d = out_%d' % (i, pos))
                    del curr_input_feature_sizes[k]
                    break

            for pos, input_feature_size in curr_input_feature_sizes:
                if input_feature_size == target_input_feature_size:
                    if pos == -1:
                        tar = 'out_begin'
                        _str = 'input_%d = torch.cat((input_%d, %s),1)' % (i, i, tar)
                    else:
                        tar = 'out_%d' % (pos)
                        _str = 'input_%d = torch.cat((input_%d, %s),1)' % (i, i, tar)

                else:
                    if pos == -1:
                        tar = 'self.sdconv_begin_end(out_begin)'
                        _str = 'input_%d = torch.cat((input_%d, %s),1)' % (i, i, tar)
                    else:
                        tar = 'self.sdconv_%d_end(out_%d)' % (pos, pos)
                        _str = 'input_%d = torch.cat((input_%d, %s),1)' % (i, i, tar)
                forward_list.append(_str)
        return forward_list

    @classmethod
    def generate_forward_list(cls, particle_a, particle_c):

        forward_list = []
        forward_list.append('out_begin = self.Hswish(self.bn_begin(self.conv_begin(x)))')
        for i, dimen in enumerate(particle_a):
            if i == 0:
                last_out_put = 'out_begin'
                _str = 'out_%d = self.conv_%d(%s)' % (i, i, last_out_put)
                forward_list.append(_str)
            else:
                bin_code = "".join([str((particle_c[i] >> y) & 1) for y in range(i, -1, -1)])
                posi = [jj for jj in range(len(bin_code)) if int(bin_code[jj]) == 1]
                if len(posi) == 1:
                    if posi[0] == 0:
                        forward_list.append('input_%d = out_begin' % (i))
                    else:
                        forward_list.append('input_%d = out_%d' % (i, posi[0] - 1))

                else:
                    forward_list = cls.generate_concat_code(particle_a, particle_c, i, bin_code, forward_list)

                if 0 <= dimen <= 31:
                    _str = 'out_%d = self.conv_%d(input_%d)' % (i, i, i)
                    forward_list.append(_str)
                else:
                    if 64 <= dimen <= 79:
                        _str = 'out_%d = F.max_pool2d(input_%d, kernel_size = 3, stride = 2, padding = 1)' % (i, i)
                    else:
                        _str = 'out_%d = F.avg_pool2d(input_%d, kernel_size = 3, stride = 2, padding = 1)' % (i, i)
                    forward_list.append(_str)

        # add final concatenate information
        forward_list = cls.generate_concat_code_outputNode(particle_a, particle_c, len(particle_a), forward_list)

        forward_list.append('out = input_%d' % (len(particle_a)))
        forward_list.append('out = self.Hswish(self.bn_end1(self.conv_end(out)))')

        forward_list.append('out = F.adaptive_avg_pool2d(out,(1,1))')
        return forward_list

    @classmethod
    def read_template(cls):
        dataset = str(cls.__read_ini_file('settings', 'dataset'))
        _path = './template/' + dataset + '.py'
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part2))

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part3))
        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, particle, curr_gen, id):
        act_funcs = ['relu', 'h_swish', 'h_swish', 'h_swish', 'h_swish']
        drop_rates = [0.0, 0.1, 0.1, 0.1, 0.1]
        # query convolution unit
        conv_name_list = []
        conv_list = []
        [particle_a, particle_c] = particle
        inchannels, outchannels, channels_end = cls.calc_in_out_channels(particle_a, particle_c)

        img_channel = cls.get_params('network', 'image_channel')
        conv_begin1 = 'self.conv_begin = nn.Conv2d(%d, %d, kernel_size=3, stride=1, padding=1, bias=False)' % (img_channel, inchannels[0])
        conv_list.append(conv_begin1)
        bn_begin1 = 'self.bn_begin = nn.BatchNorm2d(%d)' % (inchannels[0])
        conv_list.append(bn_begin1)

        for i, dimen in enumerate(particle_a):
            in_channel = inchannels[i]
            out_channel = outchannels[i]
            if 0 <= dimen <= 31:
                conv_name = 'self.conv_%d' % (i)

                if conv_name not in conv_name_list:
                    pool_sta = [1 if 64 <= particle_a[k] <= 95 else 0 for k in range(0, i)]
                    num_pool = sum(pool_sta)
                    conv_name_list.append(conv_name)
                    conv = '%s = BasicBlock(in_planes=%d, planes=%d, kernel_size=%d, expansion_rate=%d, act_func="%s", drop_connect_rate=%.1f)' % (
                    conv_name, in_channel, out_channel, kernel_sizes[num_pool], expansion_rates[num_pool],
                    act_funcs[num_pool], drop_rates[num_pool])
                    conv_list.append(conv)

        conv_end1 = 'self.conv_end = nn.Conv2d(%d, %d, kernel_size=1, stride=1, padding=1, bias=False)'%(channels_end, channels_end)
        conv_list.append(conv_end1)
        bn_end1 = 'self.bn_end1 = nn.BatchNorm2d(%d)' %(channels_end)
        conv_list.append(bn_end1)

        output_feature_sizes = cls.obtain_output_size(particle_a, particle_c)
        layer_indices = cls.obtain_final_output_layers(particle_c)
        layer_indices.append(len(particle_c) - 1)
        for idx, val in enumerate(particle_c):
            bin_code = "".join([str((val >> y) & 1) for y in range(idx, -1, -1)])
            if bin_code.count('1') > 1:
                curr_input_feature_sizes = [[j - 1, output_feature_sizes[j]] for j, bit in enumerate(bin_code) if
                                            int(bit) == 1]
                target_input_feature_size = min([k for _, k in curr_input_feature_sizes])
                for pos, input_feature_size in curr_input_feature_sizes:
                    if input_feature_size > target_input_feature_size:
                        if pos == -1:
                            sdconv = 'self.sdconv_begin_%d = nn.Conv2d(%d, %d, kernel_size=3, stride=%d, padding=1, bias=False, groups=%d)' % (
                            idx, inchannels[0],inchannels[0],
                            math.ceil(input_feature_size / target_input_feature_size),inchannels[0])
                        else:
                            sdconv = 'self.sdconv_%d_%d = nn.Conv2d(%d, %d, kernel_size=3, stride=%d, padding=1, bias=False, groups=%d)' % (
                            pos, idx, outchannels[pos], outchannels[pos],
                            math.ceil(input_feature_size / target_input_feature_size),outchannels[pos])
                        conv_list.append(sdconv)
        curr_input_feature_sizes = [[indx, output_feature_sizes[indx + 1]] for indx in layer_indices]
        target_input_feature_size = min([k for _, k in curr_input_feature_sizes])
        for pos, input_feature_size in curr_input_feature_sizes:
            if input_feature_size > target_input_feature_size:
                if pos == -1:
                    sdconv = 'self.sdconv_begin_end = nn.Conv2d(%d, %d, kernel_size=3, stride=%d, padding=1, bias=False, groups=%d)' % (
                        inchannels[0],inchannels[0],
                        math.ceil(input_feature_size / target_input_feature_size),inchannels[0])
                else:
                    sdconv = 'self.sdconv_%d_end = nn.Conv2d(%d, %d, kernel_size=3, stride=%d, padding=1, bias=False, groups=%d)' % (
                        pos, outchannels[pos], outchannels[pos],
                        math.ceil(input_feature_size / target_input_feature_size),outchannels[pos])
                conv_list.append(sdconv)

        dataset = str(cls.__read_ini_file('settings', 'dataset'))
        if dataset == 'cifar10':
            factor = 2
        elif dataset == 'cifar100':
            factor = 2
        elif dataset == 'imagenet':
            factor = 16
        fully_layer_name1 = 'self.linear1 = nn.Linear(%d, %d)' % (channels_end, factor * channels_end)
        fully_layer_name2 = 'self.dropout = nn.Dropout(p=%.1f, inplace=True)' % (0.5)
        fully_layer_name3 = 'self.bn_end2 = nn.BatchNorm1d(%d)' % (factor * channels_end)
        fully_layer_name4 = 'self.linear = nn.Linear(%d, %d)' % (factor * channels_end, cls.get_params('network', 'num_class'))

        # generate the forward part
        forward_list = cls.generate_forward_list(particle_a, particle_c)

        part1, part2, part3 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\n        %s' % ('#conv unit'))
        for s in conv_list:
            _str.append('        %s' % (s))
        _str.append('\n        %s' % ('#linear unit'))
        _str.append('        %s' % (fully_layer_name1))
        _str.append('        %s' % (fully_layer_name2))
        _str.append('        %s' % (fully_layer_name3))
        _str.append('        %s' % (fully_layer_name4))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.extend(part3)
        # print('\n'.join(_str))
        file_path = './scripts/particle%02d_%02d.py' % (curr_gen, id)
        script_file_handler = open(file_path, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        file_name = 'particle%02d_%02d'%(curr_gen, id)
        return file_name


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("EPCNAS")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warning(_str)


class GPUTools(object):
    @classmethod
    def _get_equipped_gpu_ids_and_used_gpu_info(cls):
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        equipped_gpu_ids = []
        for line_info in lines:
            if not line_info.startswith(' '):
                if 'GeForce' in line_info or 'Quadro' in line_info or 'Tesla' in line_info or 'RTX' in line_info:
                    equipped_gpu_ids.append(line_info.strip().split(' ', 4)[3])
            else:
                break

        gpu_info_list = []
        for line_no in range(len(lines) - 3, -1, -1):
            if lines[line_no].startswith('|==='):
                break
            else:
                gpu_info_list.append(lines[line_no][1:-1].strip())

        return equipped_gpu_ids, gpu_info_list

    @classmethod
    def get_available_gpu_ids(cls):
        equipped_gpu_ids, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))

        unused_gpu_ids = []
        for id_ in equipped_gpu_ids:
            if id_ not in used_gpu_ids:
                unused_gpu_ids.append(id_)
        return unused_gpu_ids

    @classmethod
    def detect_available_gpu_id(cls):
        unused_gpu_ids = cls.get_available_gpu_ids()
        if len(unused_gpu_ids) == 0:
            Log.info('GPU_QUERY-No available GPU')
            return None
        else:
            Log.info('GPU_QUERY-Available GPUs are: [%s], choose GPU#%s to use' % (','.join(unused_gpu_ids), unused_gpu_ids[0]))
            return int(unused_gpu_ids[0])

    @classmethod
    def all_gpu_available(cls):
        _, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))
        if len(used_gpu_ids) == 0:
            Log.info('GPU_QUERY-None of the GPU is occupied')
            return True
        else:
            Log.info('GPU_QUERY- GPUs [%s] are occupying' % (','.join(used_gpu_ids)))
            return False


