import argparse

parser = argparse.ArgumentParser(description='p-adic RBM')

parser.add_argument('--p', type=int, default=3,
                    help='value of p')

parser.add_argument('--l', type=int, default=3,
                    help='value of l')

parser.add_argument('--n_features', type=int, default=3**6,
                    help='number of visible')

parser.add_argument('--h_features', type=int, default=3**(7),
                    help='number of hidden')

# n_features should equal to n_components and should equal to p^{2l} for 2D images

parser.add_argument('--patch_per_image', type=int, default=16,
                    help='maximal image patches for each train image')

parser.add_argument('--patch_size', type=int, default=64,
                    help='output patch size')

parser.add_argument('--n_iterations', type=int, default=3000,
                    help='output patch size')

parser.add_argument('--save_models', action='store_false',
                    help='save all intermediate models')

parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

parser.add_argument('--k', type=int, default=1,
                    help='contrastive divergence order')

args = parser.parse_args()
