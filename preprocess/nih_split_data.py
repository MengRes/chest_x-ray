import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ========= Random seed control =========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ========= Load dataset =========
csv_path = "../files/nih14_label/Data_Entry_2017.csv"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)
print(f"Total images: {len(df)}")
print(f"Unique patients: {df['Patient ID'].nunique()}")

# ========= Patient-level split =========
unique_patient_ids = df["Patient ID"].unique()
train_ids, temp_ids = train_test_split(unique_patient_ids, test_size=0.3, random_state=SEED)
valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=SEED)

# ========= Map back to dataframe =========
train_set = df[df["Patient ID"].isin(train_ids)].reset_index(drop=True)
valid_set = df[df["Patient ID"].isin(valid_ids)].reset_index(drop=True)
test_set  = df[df["Patient ID"].isin(test_ids)].reset_index(drop=True)

print(f"[Train]  Images: {len(train_set)} | Patients: {len(train_ids)}")
print(f"[Valid]  Images: {len(valid_set)} | Patients: {len(valid_ids)}")
print(f"[Test ]  Images: {len(test_set)} | Patients: {len(test_ids)}")

# ========= Overlap check =========
def check_overlap(set1, set2, name1, name2):
    overlap = set(set1).intersection(set(set2))
    if overlap:
        print(f"[Warning] Overlap between {name1} and {name2}: {len(overlap)} patients")
    else:
        print(f"[OK] No overlap between {name1} and {name2}")

check_overlap(train_ids, valid_ids, "Train", "Valid")
check_overlap(train_ids, test_ids, "Train", "Test")
check_overlap(valid_ids, test_ids, "Valid", "Test")

# ========= Save split =========
train_set.to_csv(os.path.join(save_dir, "nih14_train_patient.csv"), index=False)
valid_set.to_csv(os.path.join(save_dir, "nih14_valid_patient.csv"), index=False)
test_set.to_csv(os.path.join(save_dir, "nih14_test_patient.csv"), index=False)