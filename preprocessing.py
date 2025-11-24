"""
preprocessing.py

MIMIC-III ADMISSIONS / ICUSTAYS / NOTEEVENTS 를 이용해서
TM-HGNN 입력용 DATA_RAW 디렉토리 구조를 만드는 전처리 스크립트.

1) data/raw/ 에 다음 3개 CSV가 있다고 가정한다.
   - ADMISSIONS.csv
   - ICUSTAYS.csv
   - NOTEEVENTS.csv

2) ICU 입실(INTIME) 기준 48시간 이내 노트만 필터링해서
   환자-episode 단위로 묶은 뒤

   data/DATA_RAW/in-hospital-mortality/ 아래에
   - train_note/listfile.csv
   - test_note/listfile.csv
   - train_note/{SUBJECT_ID}_episode{ID}.csv
   - test_note/{SUBJECT_ID}_episode{ID}.csv

   구조로 저장한다.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def build_paths():
    """
    레포 루트 기준으로 data 경로들을 생성/반환한다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    data_raw_dir = os.path.join(data_dir, "DATA_RAW")
    task_name = "in-hospital-mortality"
    task_raw_dir = os.path.join(data_raw_dir, task_name)

    # 필요 폴더 생성
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(task_raw_dir, exist_ok=True)
    os.makedirs(os.path.join(task_raw_dir, "train_note"), exist_ok=True)
    os.makedirs(os.path.join(task_raw_dir, "test_note"), exist_ok=True)

    return {
        "BASE_DIR": base_dir,
        "DATA_DIR": data_dir,
        "RAW_DIR": raw_dir,
        "DATA_RAW_DIR": data_raw_dir,
        "TASK_NAME": task_name,
        "TASK_RAW_DIR": task_raw_dir,
    }


def load_mimic_tables(raw_dir):
    """
    data/raw 에 있는 MIMIC CSV들을 읽어온다.
    파일 이름은 ADMISSIONS.csv, ICUSTAYS.csv, NOTEEVENTS.csv 로 가정.
    """

    note_path = os.path.join(raw_dir, "NOTEEVENTS.csv")
    adm_path = os.path.join(raw_dir, "ADMISSIONS.csv")
    icu_path = os.path.join(raw_dir, "ICUSTAYS.csv")

    print("=== Loading raw CSVs ===")
    print("NOTEEVENTS:", note_path)
    print("ADMISSIONS:", adm_path)
    print("ICUSTAYS :", icu_path)

    notes_raw = pd.read_csv(
        note_path,
        parse_dates=["CHARTDATE", "CHARTTIME"],
        low_memory=False,
    )

    adm_raw = pd.read_csv(
        adm_path,
        parse_dates=["ADMITTIME", "DISCHTIME"],
        low_memory=False,
    )

    icu_raw = pd.read_csv(
        icu_path,
        parse_dates=["INTIME", "OUTTIME"],
        low_memory=False,
    )

    return notes_raw, adm_raw, icu_raw


def build_final_dataset(notes_raw, adm_raw, icu_raw):
    """
    ADMISSIONS / ICUSTAYS / NOTEEVENTS 를 합쳐서
    ICU INTIME 기준 48h 내 노트만 남긴 final_dataset 생성.
    """

    # ADMISSIONS: 필요한 컬럼만 선택
    adm_cols = ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "HOSPITAL_EXPIRE_FLAG"]
    if "DEATHTIME" in adm_raw.columns:
        adm_cols.append("DEATHTIME")
    admissions = adm_raw[adm_cols].copy()

    # NOTEEVENTS: ISERROR 있으면 1인 노트 제거
    note_cols = ["SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "TEXT"]
    if "ISERROR" in notes_raw.columns:
        note_cols.insert(4, "ISERROR")  # TEXT는 마지막
    notes = notes_raw[note_cols].copy()

    if "ISERROR" in notes.columns:
        notes = notes[notes["ISERROR"].fillna(0) != 1]

    # TEXT 클리닝 (NaN -> "", 연속 공백 축소)
    notes["TEXT"] = (
        notes["TEXT"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # HADM_ID 결측 제거 + 타입 통일
    notes = notes.dropna(subset=["HADM_ID"]).copy()
    notes["HADM_ID"] = notes["HADM_ID"].astype("int64", errors="ignore")

    admissions = admissions.dropna(subset=["HADM_ID"]).copy()
    admissions["HADM_ID"] = admissions["HADM_ID"].astype("int64", errors="ignore")

    icustays = icu_raw.dropna(subset=["HADM_ID"]).copy()
    icustays["HADM_ID"] = icustays["HADM_ID"].astype("int64", errors="ignore")

    # ICU INTIME 기준 가장 첫 번째 stay만 사용
    icustays = (
        icustays
        .sort_values("INTIME")
        .drop_duplicates(subset=["SUBJECT_ID", "HADM_ID"], keep="first")
        .copy()
    )

    # 병합: notes + admissions + icustays
    df = (
        notes
        .merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner")
        .merge(icustays, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    )

    # ICU INTIME 기준 48시간 이내 노트만 남기기
    df = df.dropna(subset=["INTIME", "CHARTTIME"]).copy()

    time_mask = (
        (df["CHARTTIME"] >= df["INTIME"])
        & (df["CHARTTIME"] <= df["INTIME"] + pd.Timedelta(hours=48))
    )
    df = df[time_mask].copy()

    # 라벨 생성: HOSPITAL_EXPIRE_FLAG -> label
    df["label"] = df["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int)

    # ICU 입실 기준 경과 시간 (Hours)
    df["Hours"] = (df["CHARTTIME"] - df["INTIME"]).dt.total_seconds() / 3600.0

    final_dataset = df[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "CATEGORY",
            "TEXT",
            "CHARTTIME",
            "INTIME",
            "Hours",
            "label",
            "HOSPITAL_EXPIRE_FLAG",
        ]
    ].copy()

    print("\n=== 최종 데이터셋 샘플 ===")
    print(final_dataset.head(3))
    print(f"\n최종 행 수(=ICU 입실 후 48h 노트 수): {len(final_dataset):,}")

    print("\n라벨 분포 (%):")
    print((final_dataset["label"].value_counts(normalize=True) * 100).round(2))

    print("\nfinal_dataset.info():")
    print(final_dataset.info())

    return final_dataset


def make_episode_id(df):
    """
    HADM_ID 기준으로 episode id 부여.
    """
    df = df.copy()
    df["episode"] = df.groupby("HADM_ID").ngroup().astype(str)
    return df


def make_listfile(df, save_dir, split):
    """
    train_note / test_note 각각에 대응하는 listfile.csv 생성.
    """
    list_records = []
    for (subj, hadm, epi), g in df.groupby(["SUBJECT_ID", "HADM_ID", "episode"]):
        label = int(g["label"].iloc[0])
        list_records.append(
            {
                "patient": subj,
                "episode": f"episode{epi}",
                "y_true": label,
            }
        )

    listfile = pd.DataFrame(list_records)
    out_path = os.path.join(save_dir, f"{split}_note", "listfile.csv")
    listfile.to_csv(out_path, index=False)
    print(f"{split}_note/listfile.csv 저장 완료 ({len(listfile)}행) → {out_path}")
    return listfile


def save_episode_csvs(df, save_dir, split):
    """
    SUBJECT_ID / HADM_ID / episode 조합별로 하나의 파일 생성.

    컬럼:
    - Hours, HADM_ID, SUBJECT_ID, CATEGORY, TEXT, label
    - note_id, SENT, WORD (TM-HGNN downstream에서 사용)
    """
    base_path = os.path.join(save_dir, f"{split}_note")
    os.makedirs(base_path, exist_ok=True)

    print(f"\n[{split}] episode별 csv 저장 중 ...")
    for (subj, hadm, epi), g in tqdm(
        df.groupby(["SUBJECT_ID", "HADM_ID", "episode"]),
        desc=f"Saving {split}_note files",
    ):
        fname = f"{subj}_episode{epi}.csv"
        g_out = g[
            [
                "Hours",
                "HADM_ID",
                "SUBJECT_ID",
                "CATEGORY",
                "TEXT",
                "label",
            ]
        ].copy()

        # downstream 에서 요구하는 형태 (note_id / SENT / WORD)
        g_out["note_id"] = range(len(g_out))
        g_out["SENT"] = g_out["TEXT"].str[:50]
        g_out["WORD"] = (
            g_out["TEXT"].str.split().str[:10].apply(lambda x: " ".join(x))
        )

        g_out.to_csv(
            os.path.join(base_path, fname),
            sep="\t",
            index=False,
        )

    print(f"{split}_note 파일 저장 완료 ✅ → {base_path}")


def main():
    paths = build_paths()
    raw_dir = paths["RAW_DIR"]
    task_raw_dir = paths["TASK_RAW_DIR"]

    # 1) 원본 테이블 로드
    notes_raw, adm_raw, icu_raw = load_mimic_tables(raw_dir)

    # 2) ICU 48시간 윈도우를 적용한 final_dataset 생성
    final_dataset = build_final_dataset(notes_raw, adm_raw, icu_raw)

    print("\n=== train/test split ===")
    patients = final_dataset["SUBJECT_ID"].unique()
    train_pats, test_pats = train_test_split(
        patients, test_size=0.2, random_state=42
    )

    train_df = final_dataset[
        final_dataset["SUBJECT_ID"].isin(train_pats)
    ].copy()
    test_df = final_dataset[
        final_dataset["SUBJECT_ID"].isin(test_pats)
    ].copy()

    print(f"train 환자 수: {len(train_pats)}, test 환자 수: {len(test_pats)}")
    print(
        f"train 노트 수: {len(train_df):,}, "
        f"test 노트 수: {len(test_df):,}"
    )

    # 3) episode id 부여
    train_df = make_episode_id(train_df)
    test_df = make_episode_id(test_df)

    # 4) listfile 생성
    make_listfile(train_df, task_raw_dir, "train")
    make_listfile(test_df, task_raw_dir, "test")

    # 5) episode별 csv 생성
    save_episode_csvs(train_df, task_raw_dir, "train")
    save_episode_csvs(test_df, task_raw_dir, "test")

    print("\n✅ TM-HGNN 인풋 폴더 구조 생성 완료!")
    print(f"결과 폴더: {task_raw_dir}")


if __name__ == "__main__":
    main()
