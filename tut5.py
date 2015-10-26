

from multi_layer_perceptron import run_mlp
from pickle_data_loader import PickleDataLoader

if __name__ == '__main__':
    datasets = PickleDataLoader().load_data('mnist.pkl.gz')
    run_mlp(datasets)