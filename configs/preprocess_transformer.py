import torch
import pickle
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utill import *
class Train_Eval:

    def prepare_temporal(self, pid, model_dir):
        set_seed(pid)
        # 데이터 로딩
        data_dir = 'data/'
        model_path = os.path.join(data_dir, model_dir)
        train_path = os.path.join(model_path, 'train_temporal_transformer_dataset.pkl')
        val_path = os.path.join(model_path, 'val_temporal_transformer_dataset.pkl')
        test_path = os.path.join(model_path, 'test_temporal_transformer_dataset.pkl')

        print("Preparing datasets")
        # Transaction 데이터 로딩
        dtypes = {
            '계약날짜': 'str', '도로명': 'str', '주소': 'str',
            '우편번호': 'str', 'x': 'float64', 'y': 'float64',
            '동': 'str', '층': 'int64', '거래유형': 'str', 
            '전용면적': 'float64', '건물연식': 'int64', 
            '거래금액(만원)': 'int64'
        }
        transaction_data = pd.read_csv(os.path.join(data_dir, 'transaction_data.csv'), index_col=0, dtype=dtypes)
        df_traffic = pd.read_csv(os.path.join(data_dir, 'df_transaction_traffic.csv'), encoding='utf-8-sig', sep=',', index_col=0)
        df_amenity = pd.read_csv(os.path.join(data_dir, 'df_transaction_xy_am.csv'), encoding='utf-8-sig', sep=',', index_col=0)
        df_education = pd.read_csv(os.path.join(data_dir, 'df_transaction_xy_edu.csv'), encoding='utf-8-sig', sep=',', index_col=0)
        df_reconstruct = pd.read_csv(os.path.join(data_dir, 'df_transaction_xy_reconstruct.csv'), encoding='utf-8-sig', sep=',', index_col=0)
        df_finance = pd.read_csv(os.path.join(data_dir, 'df_finance.csv'), sep=',')
        df_finance = df_finance.rename(columns={'yearmonth': 'year_month'})

        # Merge 데이터프레임
        for df in [df_traffic, df_amenity, df_education, df_reconstruct]:
            transaction_data = pd.merge(transaction_data, df, on=['계약날짜', 'x', 'y'], how='left')

        transaction_data['계약날짜'] = pd.to_datetime(transaction_data['계약날짜'])
        transaction_data['year_month'] = transaction_data['계약날짜'].dt.year * 100 + transaction_data['계약날짜'].dt.month
        transaction_data = pd.merge(transaction_data, df_finance, on=['year_month'], how='left')
        transaction_data = transaction_data[(transaction_data['year_month'] >= 202300) & (transaction_data['year_month'] < 202400)]
        transaction_data = pd.get_dummies(transaction_data, columns=['거래유형'], dtype=float)
        transaction_data = transaction_data.rename(columns={
            '계약날짜': 'trans_date', '도로명': 'road_addr', '주소': 'addr', '우편번호': 'postal_code',
            '동': 'build_num', '층': 'floor', '거래유형_-': 'trans_type_null',
            '거래유형_중개거래': 'trans_type_indirect', '거래유형_직거래': 'trans_type_direct',
            '전용면적': 'area', '건물연식': 'build_age', '거래금액(만원)': 'price'
        })

        if 'trans_type_null' not in transaction_data.columns:
            transaction_data['trans_type_null'] = 0

        transaction_data = transaction_data[(transaction_data['price'] >= 10000) & (transaction_data['price'] <= 500000)]

        # dict_postal_data 로딩 및 전처리
        with open(os.path.join(data_dir, "dict_postal_data.pkl"), "rb") as file:
            self.dict_postal_data = pickle.load(file)

        self.preprocessed_dict_postal_data = {
            ym: df.set_index('postal_code').to_dict('index') 
            for ym, df in self.dict_postal_data.items()
        }

        # postal_node_adjacency 로딩 및 처리
        postal_node_adjacency = pd.read_csv(os.path.join(data_dir, 'postal_node_adjacency.csv'), sep=',')
        postal_node_adjacency = postal_node_adjacency.rename(columns={'우편번호': 'postal_code'})
        postal_node_adjacency['postal_code'] = postal_node_adjacency['postal_code'].apply(lambda x: str(x).zfill(5))
        transaction_data = transaction_data[transaction_data['postal_code'].isin(postal_node_adjacency['postal_code'])]
        postal_node_adjacency = postal_node_adjacency.set_index('postal_code')

        # postal_code와 인덱스 매핑
        self.postal_code_to_index = {node_id: idx for idx, node_id in enumerate(postal_node_adjacency.index)}
        self.index_to_postal_code = {idx: node_id for node_id, idx in self.postal_code_to_index.items()}

        # edge_index 준비
        edge_index = []
        for i, row in postal_node_adjacency.iterrows():
            neighbors = row[row == 1].index.values
            for neighbor in neighbors:
                edge_index.append([self.postal_code_to_index[i], self.postal_code_to_index[neighbor]])

        self.postal_code_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # node_neighbors 사전 생성
        node_neighbors = defaultdict(list)
        for src, tar in self.postal_code_edge_index.T:
            node_neighbors[int(src)].append(int(tar))

        # Train:Valid:Test 분할 70:15:15
        X = transaction_data[[
            'trans_date', 'year_month', 'road_addr', 'addr', 'postal_code', 'build_num', 'floor',
            'trans_type_null', 'trans_type_indirect', 'trans_type_direct', 'area', 'build_age',
            'interest', 'maturity', 'usdkrw', 'kospi',
            'tube_cnt_1km', 'busstop_cnt_1km', 'util_cnt_2km', 'amenity_cnt_2km', 
            'park_cnt_2km', 'util_cnt_2km', 'reconstruct_2km',
            'school_cnt_5km', 'childcare_cnt_5km'
        ]]
        y = transaction_data['price'] / 1000

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=pid)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=pid)

        print(f"X_all: {X.shape}, y_all: {y.shape}")
        print('Setup Train Dataset')
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        train_dataset = self.prepare_temporal_dataset_optimized(X_train, y_train, node_neighbors)
        print('Setup Valid Dataset')
        print(f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
        val_dataset = self.prepare_temporal_dataset_optimized(X_valid, y_valid, node_neighbors)
        print('Setup Test Dataset')
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        test_dataset = self.prepare_temporal_dataset_optimized(X_test, y_test, node_neighbors)

        # 데이터 저장
        train_dataset.to_pickle(train_path)
        val_dataset.to_pickle(val_path)
        test_dataset.to_pickle(test_path)

    def prepare_temporal_dataset_optimized(self, data_x, data_y, node_neighbors):
        postal_codes = data_x['postal_code'].values
        year_months = data_x['year_month'].values
        transaction_amounts = data_y.values
        apart_features = data_x[['floor','trans_type_null','trans_type_indirect','trans_type_direct','area','build_age','interest','maturity','usdkrw','kospi']].values.astype(np.float32)

        final_data_seq = []
        final_data_apart_features = []
        final_data_y = []

        for i in tqdm(range(len(data_x)), total=len(data_x), leave=True):
            postal_code = postal_codes[i]
            transaction_amount = transaction_amounts[i]
            year_month = year_months[i]
            year = year_month // 100
            month = year_month % 100

            intervals = []
            for j in range(11):
                offset_year = year - (j // 2)
                offset_month = month - (6 * (j % 2))
                if offset_month <= 0:
                    offset_year -= 1
                    offset_month += 12
                intervals.append(offset_year * 100 + offset_month)

            # Initialize sequence with start token
            seq = [np.array([100000,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)]

            for ym in intervals:
                if ym not in self.preprocessed_dict_postal_data:
                    continue
                ym_data = self.preprocessed_dict_postal_data[ym]
                if postal_code not in ym_data:
                    continue

                # Get features of the postal_code
                features_dict = ym_data[postal_code]
                # print(features_dict)
                features = np.array(list(features_dict.values()), dtype=np.float32)
                seq.append(features)

                # Get node index for the postal_code
                postal_code_idx = self.postal_code_to_index.get(postal_code, None)
                if postal_code_idx is None:
                    continue
                neighbors = node_neighbors.get(postal_code_idx, [])

                for neighbor_idx in neighbors:
                    neighbor_postal_code = self.index_to_postal_code.get(neighbor_idx, None)
                    if neighbor_postal_code is None or neighbor_postal_code not in ym_data:
                        continue
                    neighbor_features_dict = ym_data[neighbor_postal_code]
                    # Assuming the first element is postal_code, so skip it
                    neighbor_features = np.array(list(neighbor_features_dict.values()), dtype=np.float32)
                    seq.append(neighbor_features)

                # Append zero vector
                seq.append(np.zeros(12, dtype=np.float32))

            # Append end token
            if seq:
                seq.pop()
            seq.append(np.array([-100000,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32))
            
            final_data_seq.append(seq)
            final_data_apart_features.append(apart_features[i])
            final_data_y.append(transaction_amount)

        # 변환을 최소화하기 위해 torch 텐서는 리스트로 저장
        final_df = pd.DataFrame({
            'seq': final_data_seq,
            'apart_features': list(final_data_apart_features),
            'y': final_data_y
        })

        return final_df

if __name__ == "__main__":
    train_eval = Train_Eval()
    train_eval.prepare_temporal(0, '0_2023_2023')
