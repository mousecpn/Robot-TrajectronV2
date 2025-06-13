import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conf",
                    help="path to json config file for hyperparameters",
                    type=str,
                    default='config/config.json')

parser.add_argument("--debug",
                    help="disable all disk writing processes.",
                    action='store_true')

parser.add_argument("--preprocess_workers",
                    help="number of processes to spawn for preprocessing",
                    type=int,
                    default=0)


# Data Parameters
parser.add_argument("--data_path",
                    help="json",
                    type=str,
                    default="/home/data/data_from_root/data/data_bmi2d_w_goals_colavoid_1m.json")

parser.add_argument("--checkpoint",
                    help="the checkpoint file",
                    type=str,
                    default=None)


parser.add_argument('--device',
                    help='what device to perform training on',
                    type=str,
                    default='cuda:0')

# Training Parameters
parser.add_argument("--train_epochs",
                    help="number of iterations to train for",
                    type=int,
                    default=1)

parser.add_argument('--batch_size',
                    help='training batch size',
                    type=int,
                    default=256)


parser.add_argument('--k_eval',
                    help='how many samples to take during evaluation',
                    type=int,
                    default=25)

parser.add_argument('--seed',
                    help='manual seed to use, default is 123',
                    type=int,
                    default=123)

parser.add_argument('--eval_every',
                    help='how often to evaluate during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--map_encoding',
                    help='whether to use map encoding',
                    type=bool,
                    default=True)

args = parser.parse_args()
