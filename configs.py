import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'cs', 'facebook', 'github', 'lastfmasia'])
    # for large datasets
    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--sample_size', type=int, default=3000)  # subgraph size
    '''model'''
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gat'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--norm', action='store_true', default=0)  # apply feature norm at the beginning of the model
    '''attacker'''
    parser.add_argument('--attacker', type=int, default=1)  # 0 - our method; 1 - link teller; 2 - link stealing
    parser.add_argument('--feat', action='store_true', default=0)  # whether node features are accessible
    parser.add_argument('--targets', type=int, nargs='*', default=None)
    parser.add_argument('--sample', type=int, default=None)  # for Maui_efficient: the number of sampling times for each target node
    parser.add_argument('--combo', action='store_true', default=0)  # Maui_comb
    parser.add_argument('--combo_bar', type=float, default=2e-1)  # threshold for Maui_comb
    parser.add_argument('--efficient', action='store_true', default=0)  # for attacker 0 and 1
    parser.add_argument('--pos_type', type=int, default=0)  # posterior type for attacker 2: 0 - prediction; 1 - node features
    parser.add_argument('--topk', type=float, default=1.0)  # rate of top-k
    '''attacker execution'''
    parser.add_argument('--erase', action='store_true', default=1)  # erase previous results (start new)
    '''defense'''
    parser.add_argument('--defense', type=int, default=0)  # 1 - LapGraph; 2 - EdgeAttn
    parser.add_argument('--epsilon', type=float, default=10)
    parser.add_argument('--eval_true', action='store_true', default=0)  # whether to evaluate on the ground truth
    # under defense
    parser.add_argument('--beta', type=float, default=1.5)

    return parser.parse_args()
