import argparse

parser = argparse.ArgumentParser()

# files
parser.add_argument('--output_dir', type=str, default='./saved/', help='')
parser.add_argument('--log_file', type=str,
                    default='./saved/', help='basket_group_split')

# main hyps
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--commit', type=str, default='rand', choices=['txt', 'dw', 'rand', 'dw_i-i'])
parser.add_argument('--seq_mode', type=str, default='sku')
parser.add_argument('--model', type=str, default='GRU4Rec')
parser.add_argument('--epochs', type=int, default=50,
                    help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--batch_size', type=int, default=2048, help='batch_size')
parser.add_argument('--embedding_size', type=int,
                    default=128, help='input sise')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden sise')
parser.add_argument('--num_layers', type=int, default=1, help='gru layers')
parser.add_argument('--no_normalize', action='store_false', help='normalize the input embedding')

# others
parser.add_argument('--device', type=int, default=1, help='gpu id')
parser.add_argument('--seed', type=int, default=24, help='seed')
parser.add_argument('--early_stop', type=int, default=10, help='early stop')


args = parser.parse_args()
