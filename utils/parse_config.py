# -*- coding:utf8 -*-
# @TIME     : 2021/3/18 11:13
# @Author   : SuHao
# @File     : parse_config.py


def parse_model_config(path):
    """解析模型配置文件"""
    file = open(path, 'r')
    lines = file.read().split('\n')                 # 一次性读取所有的行
    lines = [x for x in lines if x and not x.startswith('#')]       # 删除注释行
    lines = [x.rstrip().lstrip() for x in lines]    # rstrip() 删除 string 字符串末尾的指定字符(默认为空格).
    module_defs = []
    for line in lines:
        if line.startswith('['): # "["表示一个新的网络块的开始
            module_defs.append({})          # 向列表中加入一个孔得字典
            module_defs[-1]['type'] = line[1:-1].rstrip()       # 字典创建键“type”，值为模块名称
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0          # 默认 'batch_normalize' = 0
        else:
            key, value = line.split("=")
            value = value.strip()                               # 删除不必要的空格
            module_defs[-1][key.rstrip()] = value.strip()       # 将其他参数值加入字典

    return module_defs


def parse_data_config(path):
    """解析数据配置文件"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()          # 按行读取文件
    for line in lines:
        line = line.strip()             # 于移除字符串头尾指定的字符(默认为空格或换行符)
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options