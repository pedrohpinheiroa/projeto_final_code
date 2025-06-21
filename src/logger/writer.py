import json
from time import time

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

class Writer:
    '''Responsável por criar o summary_writer e atualiza-lo a cada passo.'''

    def __init__(self):
        self.read_configs()
        self.set_configs()
        self.summary_writer = tf.summary.create_file_writer(self.log_dir + '/' + self.name)
        self.summary_writer.set_as_default()

    def read_configs(self):
        with open('configs/writer.json', 'r') as file:
            self.configs = json.load(file)

    def set_configs(self):
        self.log_dir = self.configs['log_dir']
        self.name = str(time())
        
    def write_scalar(self, tag, value, episode):
        '''Escreve um valor escalar no summary.'''
        with self.summary_writer.as_default():
            tf.summary.scalar(tag, value, step=episode)
            self.summary_writer.flush()
    
    def write_hyperparameters(self, hyperparameters):
        '''Escreve os hiperparâmetros no summary.'''
        with self.summary_writer.as_default():
            hp.hparams(hyperparameters)
            self.summary_writer.flush()