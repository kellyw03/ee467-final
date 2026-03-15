import numpy as np 
import pandas as pd
from pathlib import Path
import tarfile
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

INPUT_DATA_PATH = "data/all_attack_benign_samples.tar.xz" 
ATTACK_PATH = "data/all_attack_benign_samples/attack_data/attack_samples_1sec.csv.tar.xz"
BENIGN_PATH = "data/all_attack_benign_samples/benign_data/benign_samples_1sec.csv.tar.xz"
REMOVE_COLS = ['label1', 'label2','label3', 'label4', 'device_name', 'device_mac', 'label_full', 'timestamp_start', 'timestamp','timestamp_end']
RAND_STATE = 67

# load data
def read_tar_xz_csv(path):
    rows = []
    total = 0
    with tarfile.open(path, mode="r:xz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".csv")]
        
        csv_member = members[0]
        f = tar.extractfile(csv_member)
        
        for chunk in pd.read_csv(f, chunksize=50000):
            rows.append(chunk)
            total += len(chunk)

    return pd.concat(rows)

# loads and returns main data
def get_data():
    attack_pd = read_tar_xz_csv(ATTACK_PATH)
    benign_pd = read_tar_xz_csv(BENIGN_PATH)
    comb_df = pd.concat([attack_pd, benign_pd])
    return comb_df

# standardize data
def standardize(data):
    scaled = preprocessing.RobustScaler().fit_transform(data)
    return scaled


# main pipeline for loading, combining, and removing data
# smote_select: boolean to use SMOTE imbalance handling
# sample_frac: take fraction of sample from dataset
# sample_n: take number of sample from dataset
def preprocess(smote_select=False, sample_frac=None, sample_n=None):
    comb_df = get_data()
    labels = comb_df['label2'].copy()

    # add new features
    comb_df['timestamp_start'] = pd.to_datetime(comb_df['timestamp_start'])
    comb_df['timestamp_end'] = pd.to_datetime(comb_df['timestamp_end'])
    comb_df = add_features(comb_df)

    # Keep numeric data for first evaluation, deal with categorical data later
    # remove id labels - only concerned with main classification
    comb_df.drop(REMOVE_COLS, axis=1, inplace=True)
    numeric_df = comb_df.select_dtypes(include=["number"])

        # Count NaNs in each column
    nan_counts = comb_df.isna().sum()

    # Show only columns that actually have NaNs
    print(nan_counts[nan_counts > 0])

    # optional downsampling before split
    if sample_frac is not None:
      numeric_df, _, labels, _ = train_test_split(
        numeric_df,
        labels,
        train_size=sample_frac,
        random_state=RAND_STATE,
        stratify=labels
    )

    if sample_n is not None:
      numeric_df, _, labels, _ = train_test_split(
        numeric_df,
        labels,
        train_size=sample_n,
        random_state=RAND_STATE,
        stratify=labels
    )

    # split w/ stratify
    x_train, x_test, y_train, y_test = train_test_split(numeric_df, labels, test_size=0.2, random_state=RAND_STATE, shuffle=True, stratify=labels)

    # optional: SMOTE imbalance
    if smote_select:
        smote_instance = SMOTE(random_state=RAND_STATE)
        x_train, y_train = smote_instance.fit_resample(x_train, y_train)

    return x_train, x_test, y_train, y_test

def add_features(data):
    data['traffic_ratio'] = data['network_packets_src_count'] / data['network_packets_dst_count'].replace(0, 1e-6)
    data['port_ratio'] = data['network_ports_src_count'] / data['network_ports_dst_count']
    data['ip_coverage'] = data['network_ips_dst_count'] / data["network_ips_all_count"]
    data['port_coverage'] = data['network_ports_dst_count'] / data["network_ports_all_count"]
    data['fragmentation'] = data["network_fragmented-packets"] / data["network_packets_all_count"]
    data['failed_conn_attempts'] = data['network_tcp-flags-rst_count'] / data['network_packets_all_count']

    data[['port_ratio','ip_coverage','port_coverage','fragmentation','failed_conn_attempts']] = \
    data[['port_ratio','ip_coverage','port_coverage','fragmentation','failed_conn_attempts']].fillna(0)

    data['packet_rate'] = data['network_packets_all_count'] / (data['timestamp_end'] - data['timestamp_start']).dt.total_seconds()
    return data