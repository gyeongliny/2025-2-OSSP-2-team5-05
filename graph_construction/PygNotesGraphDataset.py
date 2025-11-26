import os
import os.path as osp
import sys
import torch
from torch_geometric.data import InMemoryDataset

# 현재 파일 : project_root/graph_construction/PygNotesGraphDataset.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREP_NOTES_DIR = osp.join(CURRENT_DIR, 'prepare_notes')

if PREP_NOTES_DIR not in sys.path:
    sys.path.append(PREP_NOTES_DIR)

from ConstructDatasetByNotes import ConstructDatasetByNotes


# ====== 프로젝트 루트 기준 기본 경로 ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = osp.join(BASE_DIR, "data")
IMDB_PATH = osp.join(DATA_DIR, "IMDB_HCUT")
PRE_PATH = osp.join(DATA_DIR, "DATA_PRE")
RAW_PATH = osp.join(DATA_DIR, "DATA_RAW")


class PygNotesGraphDataset(InMemoryDataset):
    """
    preprocess를 수행하여 *.pt 파일을 생성하는 Dataset
    (train/test 각각 1번만 실행)
    """
    def __init__(self, name, split, tokenizer, dictionary, pre_path, data_type='hyper',
                 transform=None, pre_transform=None):

        # IMDB_HCUT/in-hospital-mortality/<tokenizer>/
        self.imdb_path = osp.join(IMDB_PATH, name, tokenizer)

        self.name = name
        self.split = split              # train / test
        self.tokenizer = tokenizer      # clinicalbert / gatortron / word2vec
        self.dictionary = dictionary
        self.data_type = data_type
        self.pre_path = pre_path

        # processed_dir = imdb_path
        super().__init__(osp.join(self.imdb_path, f"{self.split}_{self.data_type}"),
                         transform, pre_transform)

        # 이미 생성된 *.pt 파일 로드
        # 🔥 PyTorch 2.6 이상에서는 weights_only 기본값이 True라서
        #    여기서 반드시 weights_only=False 를 지정해줘야 함.
        self.data, self.slices = torch.load(
            osp.join(self.processed_dir, self.processed_file_names),
            weights_only=False
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        # IMDB_HCUT/in-hospital-mortality/<tokenizer>
        return osp.join(self.imdb_path)

    @property
    def processed_file_names(self):
        # 예: train_hyper/train_hyper_gatortron.pt
        if self.data_type == "hyper":
            return f"{self.split}_{self.data_type}/{self.split}_{self.data_type}_{self.tokenizer}.pt"
        else:
            return f"{self.split}.pt"

    def process(self):
        """
        최초 실행 시:
        hypergraph 리스트 생성 → *.pt 파일로 저장
        """

        # tokenizer 전달 추가됨 !!!
        cdbn = ConstructDatasetByNotes(
            pre_path=self.pre_path,
            split=self.split,
            dictionary=self.dictionary,
            task=self.name,
            tokenizer=self.tokenizer
        )

        # train split 첫 실행 시 카테고리 파일 생성
        if self.split == "train":
            cdbn.create_all_cats(path=self.pre_path)
            print("categories.txt created")

        if self.data_type == "hyper":
            print("Data Type : hyper")
            data_list = cdbn.construct_hypergraph_datalist()
        else:
            raise ValueError("data_type must be {hyper}")

        print("\n<Collate Data List...>")
        data, slices = self.collate(data_list)

        print("\n<Collate Done, Start Saving...>")
        save_path = osp.join(self.processed_dir, self.processed_file_names)
        os.makedirs(osp.dirname(save_path), exist_ok=True)

        torch.save((data, slices), save_path)

        print("Saving Done")
        print("<<Created", self.split, "Cutoff HyperGraph Dataset>>")


class Load_PygNotesGraphDataset(InMemoryDataset):
    """
    이미 생성된 *.pt 파일을 로드하여 학습 시 사용하는 Dataset
    """
    def __init__(self, name, split, tokenizer, dictionary,
                 data_type='hyper', transform=None, pre_transform=None):

        self.imdb_path = osp.join(IMDB_PATH, name, tokenizer)

        self.name = name
        self.split = split
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.data_type = data_type
        self.pre_path = PRE_PATH

        super().__init__(self.imdb_path, transform, pre_transform)

        # 🔥 여기서도 마찬가지로 weights_only=False 필수
        self.data, self.slices = torch.load(
            osp.join(self.processed_dir, self.processed_file_names),
            weights_only=False
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return self.imdb_path

    @property
    def processed_file_names(self):
        if self.data_type == "hyper":
            return f"{self.split}_{self.data_type}/{self.split}_{self.data_type}_{self.tokenizer}.pt"
        else:
            return f"{self.split}.pt"
