import sys
import os

# 현재 파일 위치: project_root/graph_construction/LoadData.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREP_NOTES_DIR = os.path.join(CURRENT_DIR, 'prepare_notes')

# prepare_notes 폴더를 import 경로에 추가
if PREP_NOTES_DIR not in sys.path:
    sys.path.append(PREP_NOTES_DIR)

from PygNotesGraphDataset import Load_PygNotesGraphDataset as PNGD
from embedding_utils import Handle_data
from torch_geometric.data import DataLoader
import torch
import numpy as np
import argparse
import pandas as pd


class LoadPaitentData():
    """
    TM-HGNN용 환자 단위 데이터 로더.

    - name       : task 이름 (예: 'in-hospital-mortality')
    - type       : Handle_data 에 전달할 type (예: 'cutoff')
    - data_type  : 'hyper' 등 (hypergraph 기준)
    - tokenizer  : 'clinicalbert', 'gatortron', 'word2vec' 등
    """
    def __init__(self, name, type, data_type, tokenizer='gatortron'):
        self.name = name
        self.type = type
        self.data_type = data_type

        # 프로젝트 루트 기준 vocab.txt 경로 설정
        # __file__ = project_root/graph_construction/LoadData.py
        # -> dirname 1번: graph_construction/
        # -> dirname 2번: project_root/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vocab_path = os.path.join(base_dir, 'data', 'DATA_RAW', 'root', 'vocab.txt')

        self.dictionary = open(vocab_path, encoding="utf-8").read().split()
        self.tokenizer = tokenizer

        super(LoadPaitentData, self).__init__()
        self.train_set = []
        self.val_set = []
        self.test_set = []

    def split_train_val_data(self, seed, ratio):
        """
        train_set 을 비율(ratio)대로 train/val로 나누는 함수.
        y_p 분포에 맞춰 간단하게 섞어서 split.
        """
        np.random.seed(seed)

        cs = pd.DataFrame(self.train_set.data.y_p.tolist())[0].value_counts().to_dict()
        train_id = np.arange(sum(cs.values())).tolist()
        print("len train_id:", len(train_id))
        print("len train_set:", len(self.train_set))

        np.random.shuffle(train_id)
        n_train = int(np.round(len(train_id) * ratio))

        self.val_set = self.train_set[train_id[n_train:]]
        self.train_set = self.train_set[train_id[:n_train]]
        return self.train_set, self.val_set

    def get_train_test(self, batch_size, seed, ratio=0.8):
        """
        데이터셋 로드 + train/val/test DataLoader 생성.

        - batch_size : train batch size
        - seed       : split 시드
        - ratio      : train:val 비율 (예: 0.8 → 80% train, 20% val)
        """

        print('load test data...')
        self.test_set = PNGD(
            name=self.name,
            split='test',
            tokenizer=self.tokenizer,
            dictionary=self.dictionary,
            data_type=self.data_type,
            transform=Handle_data(self.type, self.tokenizer)
        )
        # 전체 test 데이터를 한번 훑어서 제대로 로드됐는지 확인 (원 코드 유지)
        _ = [data for data in self.test_set]

        print('load train data...')
        self.train_set = PNGD(
            name=self.name,
            split='train',
            tokenizer=self.tokenizer,
            dictionary=self.dictionary,
            data_type=self.data_type,
            transform=Handle_data(self.type, self.tokenizer)
        )
        print('Train Dataset: {}'.format(len(self.train_set)))

        # train/val split
        self.train_set, self.val_set = self.split_train_val_data(seed, ratio)

        # y_p 클래스 수가 세 split에서 모두 같은지 확인
        assert self.val_set.data.y_p.unique().size(0) == \
               self.train_set.data.y_p.unique().size(0) == \
               self.test_set.data.y_p.unique().size(0)

        print('Train Dataset: {}'.format(len(self.train_set)))
        print('Val Dataset: {}'.format(len(self.val_set)))
        print('Test Dataset: {}'.format(len(self.test_set)))

        num_class = self.val_set.data.y_p.unique().size(0)

        follow_batch = ['x']  # Handle_data에서 data.x 생성하므로 'x' 기준으로 follow_batch

        train_loader = DataLoader(
            self.train_set[:],
            batch_size=batch_size,
            follow_batch=follow_batch,
            shuffle=True
        )
        val_loader = DataLoader(
            self.val_set[:],
            batch_size=10,
            follow_batch=follow_batch,
            shuffle=True
        )
        test_loader = DataLoader(
            self.test_set[:],
            batch_size=1,
            follow_batch=follow_batch,
            shuffle=False
        )

        return train_loader, val_loader, test_loader, num_class


def label_distribution(data_set):
    """
    y_p 라벨 분포 출력용 유틸 (필요하면 사용).
    """
    y_1 = 0
    y_0 = 0
    for d in data_set:
        if d.y_p == torch.tensor([1]):
            y_1 += 1
        else:
            y_0 += 1
    print('#y_1: ', str(y_1), ';#y_0: ', str(y_0))
    return y_1 + y_0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GNN for EHR")
    parser.add_argument('--name', type=str, default='in-hospital-mortality')
    parser.add_argument('--type', type=str, default='cutoff')
    parser.add_argument('--data_type', type=str, default='hyper')
    # 🔥 기본 tokenizer를 GatorTronS 기반으로 사용
    parser.add_argument('--tokenizer', type=str, default='gatortron')

    args, _ = parser.parse_known_args()
    loader = LoadPaitentData(
        name=args.name,
        type=args.type,
        tokenizer=args.tokenizer,
        data_type=args.data_type
    )
    train_loader, val_loader, test_loader, n_class = loader.get_train_test(
        batch_size=1,
        seed=2
    )
