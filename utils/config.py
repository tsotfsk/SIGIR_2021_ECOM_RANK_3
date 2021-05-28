import argparse

parser = argparse.ArgumentParser()

# files
parser.add_argument('--output_dir', type=str, default='./saved/', help='')
parser.add_argument('--log_fire', type=str,
                    default='./saved/', help='basket_group_split')

# main hyps
parser.add_argument('--evalaute', action='store_true')
parser.add_argument('--model', type=str, default='GRU4Rec')
parser.add_argument('--epochs', type=int, default=50,
                    help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
parser.add_argument('--dropout_prob', type=float, default=0.3, help='dropout')
parser.add_argument('--embedding_size', type=int,
                    default=64, help='input sise')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden sise')
parser.add_argument('--num_layers', type=int, default=1, help='gru num_layer')

# others
parser.add_argument('--device', type=int, default=0, help='GPU_ID')
parser.add_argument('--seed', type=int, default=24, help='seed')
parser.add_argument('--early_stop', type=int, default=10, help='gru num_layer')


args = parser.parse_args()
