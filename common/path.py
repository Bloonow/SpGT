import os

ROOT_PATH = os.path.join(os.path.expanduser('~'), 'SpGT')
CONFIG_PATH = os.path.join(ROOT_PATH, 'config')
EXTENSION_PATH = os.path.join(ROOT_PATH, 'extension')

STORAGE_PATH = os.path.join(ROOT_PATH, 'storage')
DATA_PATH = os.path.join(STORAGE_PATH, 'data')
MODEL_PATH = os.path.join(STORAGE_PATH, 'model')
EVALUATION_PATH = os.path.join(STORAGE_PATH, 'evaluation')
VISUALIZATION_PATH = os.path.join(STORAGE_PATH, 'visualization')
