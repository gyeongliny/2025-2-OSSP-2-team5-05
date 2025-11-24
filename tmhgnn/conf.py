import sys, os
import argparse

def sys_append(a, a_1):
    sys.path.append(os.path.join(a, a_1))

def return_paths(a):
    IMDB_PATH = os.path.join(a, './data/IMDB_HCUT')
    PRE_PATH = os.path.join(a, './data/DATA_PRE')
    RAW_PATH = os.path.join(a, './data/DATA_RAW')
    return IMDB_PATH, PRE_PATH, RAW_PATH


def user_config():
    """
    프로젝트 루트 기준으로 graph_construction 경로를 sys.path에 추가해주는 함수.
    train.py에서 맨 처음 호출됨.
    """
    # 현재 파일(conf.py) 기준: tmhgnn/conf.py
    # -> 상위 폴더: tmhgnn/
    # -> 다시 상위 폴더: 프로젝트 루트(TM-HGNN)
    base_dir = os.path.dirname(os.path.abspath(__file__))  # tmhgnn/
    base_dir = os.path.dirname(base_dir)                   # project root

    graph_dir = os.path.join(base_dir, 'graph_construction')
    if graph_dir not in sys.path:
        sys.path.append(graph_dir)


### Configs ###
def parse_arguments():
    parser = argparse.ArgumentParser()

    # DataLoader & Model
    parser.add_argument('-d', '--dload', default='multi_hyper',
                        help="Dataloader type (default: multi_hyper)")
    parser.add_argument('-m', '--model', default='TM_HGNN',
                        help="Model name in net.py (default: TM_HGNN)")

    # Output path for checkpoints
    parser.add_argument('-o', '--output',
                        default='./tmhgnn/TM_HGNN.pth',
                        help="Path to save best model checkpoint")

    # Save results (pickle with pred/label)
    parser.add_argument('-e', '--exp-name',
                        default='./tmhgnn/results',
                        help="Base path for saving result pickle")

    # Task
    parser.add_argument('-t', '--task',
                        default='in-hospital-mortality', type=str,
                        help="Task name (default: in-hospital-mortality)")

    # Data / tokenizer
    # clinicalbert, gatortron, word2vec 등으로 확장해서 사용할 예정
    parser.add_argument('--tokenizer',
                        default='gatortron', type=str,
                        help="Embedding type / tokenizer name "
                             "(e.g., clinicalbert, gatortron, word2vec)")
    parser.add_argument('--dtype',
                        default='hyper', type=str,
                        help="Data type (default: hyper for hypergraphs)")

    # Training
    parser.add_argument('-b', '--bsz',
                        default=32, type=int,
                        help="Batch size")
    parser.add_argument('--optimizer',
                        default='Adam', type=str,
                        help="Optimizer name (Adam, AdamW, ...)")
    parser.add_argument('--init-lr',
                        default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--weight-decay',
                        default=0.0, type=float,
                        help="Weight decay")
    parser.add_argument('--loss',
                        default='BCEWithLogitsLoss', type=str,
                        help="Loss function name")
    parser.add_argument('--num-epochs',
                        default=50, type=int,
                        help="Number of training epochs")

    # Model hyperparameters
    parser.add_argument('--layers',
                        default=3, type=int,
                        help="Number of TM-HGNN layers")
    parser.add_argument('--heads-1',
                        default=8, type=int,
                        help="Number of heads in first layer")
    parser.add_argument('--heads-2',
                        default=8, type=int,
                        help="Number of heads in second layer")
    parser.add_argument('--hidden-channels',
                        default=8, type=int,
                        help="Hidden channels per head")

    # 이 값은 train.py에서 데이터 기반으로 자동으로 대체됨
    parser.add_argument('--node-features',
                        default=104, type=int,
                        help="Node feature dimension (will be auto-inferred)")

    # Seed
    parser.add_argument('--seed',
                        default=50, type=int,
                        help="Random seed")

    return parser.parse_args()
