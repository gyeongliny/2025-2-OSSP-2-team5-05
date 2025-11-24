from __future__ import absolute_import, print_function
import os
import argparse
import pandas as pd
from tqdm import tqdm


def note_hyper(p_df: pd.DataFrame) -> pd.DataFrame | str:
    """
    한 patient의 note들을 hyper-edge용 row로 풀어주는 함수.

    출력 컬럼:
      Hours, HADM_ID, SUBJECT_ID, WORD, SENT, note_id, CATEGORY
    """
    dfs = []

    # TEXT 컬럼 이름이 상황에 따라 다를 수 있어서 둘 다 지원
    if "fixed TEXT" in p_df.columns:
        text_col = "fixed TEXT"
    elif "TEXT" in p_df.columns:
        text_col = "TEXT"
    else:
        raise ValueError("텍스트 컬럼('fixed TEXT' 또는 'TEXT')을 찾을 수 없습니다.")

    note_id = 0
    for i, note in enumerate(p_df[text_col]):
        hours = p_df.iloc[i]["Hours"]
        category = p_df.iloc[i]["CATEGORY"]
        hadm_id = p_df.iloc[i]["HADM_ID"]
        subject_id = p_df.iloc[i]["SUBJECT_ID"]

        # 노트 한 개를 문장 단위로 나누고, 다시 단어 단위로 펼치기
        sents = str(note).split("\n")
        sent_id = 0
        for sent in sents:
            item_list = str(sent).split()
            for item in item_list:
                dfs.append(
                    [
                        hours,
                        hadm_id,
                        subject_id,
                        item,
                        sent_id,
                        note_id,
                        category,
                    ]
                )
            sent_id += 1
        note_id += 1

    if len(dfs) == 0:
        return ""

    dfs = pd.DataFrame(
        dfs,
        columns=[
            "Hours",
            "HADM_ID",
            "SUBJECT_ID",
            "WORD",
            "SENT",
            "note_id",
            "CATEGORY",
        ],
    )
    return dfs


def create_hyper_df(partition: str, action: str, args) -> None:
    """
    train/test 각각에 대해 *_note 폴더의 episode csv를 읽어서
    *_hyper 폴더에 hypergraph용 csv로 변환.
    """
    output_dir = os.path.join(args.pre_path, args.task, partition + "_hyper")
    split = partition + "_note"

    if action != "make":
        return

    os.makedirs(output_dir, exist_ok=True)

    note_dir = os.path.join(args.raw_path, args.task, split)
    patients = [
        f for f in os.listdir(note_dir)
        if "episode" in f and f.endswith(".csv")
    ]

    for patient in tqdm(
        patients,
        desc=f"Iterating over patients in {args.raw_path}/{args.task}/{split}",
    ):
        p_df = pd.read_csv(
            os.path.join(note_dir, patient),
            sep="\t",
            engine="python",
            header=0,
        )

        p_hyper_df = note_hyper(p_df)
        if isinstance(p_hyper_df, str) or len(p_hyper_df) == 0:
            print(f"--> Warning: No nodes in patient {patient}!!")
            continue

        p_hyper_df.to_csv(
            os.path.join(output_dir, patient),
            sep="\t",
            index=False,
        )

    print(
        f"{partition} hyper samples complete! "
        f"please check in {output_dir}"
    )


if __name__ == "__main__":
    # 프로젝트 루트 기준 상대 경로를 기본값으로 사용
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DEFAULT_RAW_PATH = os.path.join(DATA_DIR, "DATA_RAW")
    DEFAULT_PRE_PATH = os.path.join(DATA_DIR, "DATA_PRE")

    parser = argparse.ArgumentParser(
        description="Create hypergraph dataframe for in-hospital mortality prediction task."
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default=DEFAULT_RAW_PATH,
        help="Directory where the cleaned data is stored (Input).",
    )
    parser.add_argument(
        "--pre_path",
        type=str,
        default=DEFAULT_PRE_PATH,
        help="Directory where the processed data should be stored (Output).",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default="100",
        help="input dimension (legacy arg, currently unused).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="in-hospital-mortality",
        help="task name: [in-hospital-mortality]",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="clinicalbert",
        help="tokenizer name (word2vec / clinicalbert / gpt 등, 현재 파일에서는 직접 사용 X).",
    )
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--partition", type=str, default="test")
    parser.add_argument("--action", type=str, default="make")

    args, _ = parser.parse_known_args()

    print("Creating train HyperSamples...")
    create_hyper_df("train", args.action, args)
    print("Creating test HyperSamples...")
    create_hyper_df("test", args.action, args)
