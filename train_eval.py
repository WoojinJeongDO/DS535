import datetime
import os
import math
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from algo_common import *
from utill import *


class Train_Eval:
    def __init__(self,config,device):
        self.config = config
        self.device = device

    def prepare(self,algs,pid,model_dir,start_epoch):
        set_seed(pid)
        #3.1 Data Loading
        
        data_dir = 'data/'
        data_path = f'data/{model_dir}'
        train_path = f'data/{model_dir}/train_dataset.pkl'
        val_path  = f'data/{model_dir}/val_dataset.pkl'
        test_path  = f'data/{model_dir}/test_dataset.pkl'
        if dataset_exists(train_path) & dataset_exists(val_path) & dataset_exists(test_path):
            #3.1 Load dataset
            print("Loading datasets from saved files...")
            with open(train_path, 'rb') as f:
                train_dataset = pickle.load(f)

            with open(val_path, 'rb') as f:
                val_dataset = pickle.load(f)

            with open(test_path, 'rb') as f:
                test_dataset = pickle.load(f)
        else:
            if not dataset_exists(data_path):
                os.mkdir(data_path)
            print("Preparing datasets")
            #3.1.1 Load Transaction data
            transaction_data = pd.read_csv(os.path.join(data_dir, 'transaction_data.csv'),index_col = 0,
                            dtype={'계약날짜': 'str', '도로명': 'str', '주소': 'str',
                                    '우편번호': 'str', 'x': 'float64', 'y':'float64',
                                    '동':'str','층': 'int64','거래유형':'str','전용면적':'float64',
                                    '건물연식':'int64','거래금액(만원)':'int64'})
            df_traffic = pd.read_csv(os.path.join(data_dir, 'df_transaction_traffic.csv'), encoding='utf-8-sig', sep=',', index_col=0)
            df_amenity = pd.read_csv(os.path.join(data_dir, 'df_transaction_xy_am.csv'), encoding='utf-8-sig', sep=',', index_col=0)
            df_education = pd.read_csv(os.path.join(data_dir, 'df_transaction_xy_edu.csv'), encoding='utf-8-sig', sep=',', index_col=0)
            df_reconstruct = pd.read_csv(os.path.join(data_dir, 'df_transaction_xy_reconstruct.csv'), encoding='utf-8-sig', sep=',', index_col=0)
            df_finance = pd.read_csv(os.path.join(data_dir, 'df_finance.csv'),  sep=',')
            df_finance = df_finance.rename(columns ={'yearmonth':'year_month'})
            for df in [df_traffic, df_amenity, df_education,df_reconstruct]:
                transaction_data = pd.merge(transaction_data, df, on=['계약날짜','x','y'], how='left')
            transaction_data['계약날짜'] = pd.to_datetime(transaction_data['계약날짜'])
            transaction_data['year_month'] = transaction_data['계약날짜'].dt.year*100+transaction_data['계약날짜'].dt.month
            transaction_data = pd.merge(transaction_data, df_finance, on=['year_month'], how='left')
            transaction_data = transaction_data.loc[(int(self.config.years[0])*100<=transaction_data['year_month'])& (transaction_data['year_month']<(int(self.config.years[-1])+1)*100)]
            transaction_data = pd.get_dummies(transaction_data, columns = ['거래유형'],dtype=float)
            transaction_data = transaction_data.rename(columns ={'계약날짜':'trans_date','도로명':'road_addr','주소':'addr','우편번호':'postal_code',
                                                                '동':'build_num','층':'floor','거래유형_-':'trans_type_null','거래유형_중개거래':'trans_type_indirect'
                                                                ,'거래유형_직거래':'trans_type_direct','전용면적':'area','건물연식':'build_age','거래금액(만원)':'price'                                                      
                                                                })
            if 'trans_type_null' not in transaction_data.columns:
                transaction_data['trans_type_null'] = 0
            transaction_data = transaction_data[(transaction_data['price']>=10000)&(transaction_data['price']<=500000)]
            #3.1.3 Load dict_postal_data
            with open(os.path.join(data_dir, "dict_postal_data.pkl"), "rb")  as file:
                self.dict_postal_data = pickle.load(file)
                
            postal_node_adjacency = pd.read_csv(os.path.join(data_dir, 'postal_node_adjacency.csv'), sep=',')
            postal_node_adjacency = postal_node_adjacency.rename(columns={'우편번호':'postal_code'})
            postal_node_adjacency['postal_code'] = postal_node_adjacency['postal_code'].apply(lambda x: str(x).zfill(5))
            transaction_data = transaction_data[transaction_data['postal_code'].isin(postal_node_adjacency['postal_code'])]
            postal_node_adjacency = postal_node_adjacency.set_index(keys='postal_code')
            node_id_to_index = {node_id: idx for idx, node_id in enumerate(postal_node_adjacency.index)}
            # Prepare edge list for the graph
            edge_index = []
            for i, row in postal_node_adjacency.iterrows():
                neighbors = row[row == 1].index.values
                for neighbor in neighbors:
                    # Map the node ID and neighbor ID to integer indices
                    edge_index.append([node_id_to_index[i], node_id_to_index[neighbor]])

            # Convert the edge list to a tensor
            self.postal_code_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            #3.1.3 Load dict_addr_data
            """
            with open(os.path.join(data_dir, "dict_road_addr_data.pkl"), "rb")  as file:
                dict_road_addr_data = pickle.load(file)
            road_addr_node_adjacency = pd.read_csv(os.path.join(data_dir, 'road_addr_node_adjacency.csv'), sep=',')
            """
            #3.2 Train:Valid:Test spilit 70:15:15
            X = transaction_data[['trans_date','year_month', 'road_addr','addr','postal_code','build_num','floor',
                                'trans_type_null','trans_type_indirect','trans_type_direct','area','build_age',
                                'interest','maturity','usdkrw','kospi',
                                'tube_cnt_1km','busstop_cnt_1km','util_cnt_2km','amenity_cnt_2km','park_cnt_2km','util_cnt_2km','reconstruct_2km',
                                'school_cnt_5km','childcare_cnt_5km']]
            
            #y = np.log10(transaction_data['price'])
            y = transaction_data['price']/1000
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=pid)
            X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=pid)

            print(f"X_all: {X.shape}, y_all: {y.shape}")
            
            # Prepare datasets
            print('Setup Train Dataset')
            print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            train_dataset = self.prepare_dataset(X_train, y_train)
            print('Setup Valid Dataset')
            print(f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
            val_dataset = self.prepare_dataset(X_valid, y_valid)
            print('Setup Test Dataset')
            print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            test_dataset = self.prepare_dataset(X_test,y_test)
            with open(train_path, 'wb') as f:
                pickle.dump(train_dataset, f)
                
            with open(val_path, 'wb') as f:
                pickle.dump(val_dataset, f)
                
            with open(test_path, 'wb') as f:
                pickle.dump(test_dataset, f)

        self.batch_size = self.config.batch_size
        # Prepare DataLoader for batch processing
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        #set_algorithm
        for alg in algs:
            alg.set_algo(model_dir,start_epoch)
    
    def prepare_temporal(self,algs,pid,model_dir,start_epoch):
        set_seed(pid)
        #3.1 Data Loading
        data_dir = 'data/'
        data_path = f'data/{model_dir}'
        train_path = f'data/{model_dir}/train_temporal_dataset.pkl'
        val_path  = f'data/{model_dir}/val_temporal_dataset.pkl'
        test_path  = f'data/{model_dir}/test_temporal_dataset.pkl'
        if dataset_exists(train_path) & dataset_exists(val_path) & dataset_exists(test_path):
            #3.1 Load dataset
            print("Loading datasets from saved files...")
            with open(train_path, 'rb') as f:
                train_dataset = pickle.load(f)

            with open(val_path, 'rb') as f:
                val_dataset = pickle.load(f)

            with open(test_path, 'rb') as f:
                test_dataset = pickle.load(f)
        else:
            if not dataset_exists(data_path):
                os.mkdir(data_path)
            print("Preparing datasets")
            #3.1.1 Load Transaction data
            transaction_data = pd.read_csv(os.path.join(data_dir, 'transaction_data.csv'),index_col = 0,
                            dtype={'계약날짜': 'str', '도로명': 'str', '주소': 'str',
                                    '우편번호': 'str', 'x': 'float64', 'y':'float64',
                                    '동':'str','층': 'int64','거래유형':'str','전용면적':'float64',
                                    '건물연식':'int64','거래금액(만원)':'int64'})
            df_traffic = pd.read_csv(os.path.join(data_dir, 'df_transaction_traffic.csv'), encoding='utf-8-sig', sep=',', index_col=0)
            df_amenity = pd.read_csv(os.path.join(data_dir, 'df_transaction_xy_am.csv'), encoding='utf-8-sig', sep=',', index_col=0)
            df_education = pd.read_csv(os.path.join(data_dir, 'df_transaction_xy_edu.csv'), encoding='utf-8-sig', sep=',', index_col=0)
            df_reconstruct = pd.read_csv(os.path.join(data_dir, 'df_transaction_xy_reconstruct.csv'), encoding='utf-8-sig', sep=',', index_col=0)
            df_finance = pd.read_csv(os.path.join(data_dir, 'df_finance.csv'),  sep=',')
            df_finance = df_finance.rename(columns ={'yearmonth':'year_month'})
            for df in [df_traffic, df_amenity, df_education,df_reconstruct]:
                transaction_data = pd.merge(transaction_data, df, on=['계약날짜','x','y'], how='left')
            transaction_data['계약날짜'] = pd.to_datetime(transaction_data['계약날짜'])
            transaction_data['year_month'] = transaction_data['계약날짜'].dt.year*100+transaction_data['계약날짜'].dt.month
            transaction_data = pd.merge(transaction_data, df_finance, on=['year_month'], how='left')
            transaction_data = transaction_data.loc[(int(self.config.years[0])*100<=transaction_data['year_month'])& (transaction_data['year_month']<(int(self.config.years[-1])+1)*100)]
            transaction_data = pd.get_dummies(transaction_data, columns = ['거래유형'],dtype=float)
            transaction_data = transaction_data.rename(columns ={'계약날짜':'trans_date','도로명':'road_addr','주소':'addr','우편번호':'postal_code',
                                                                '동':'build_num','층':'floor','거래유형_-':'trans_type_null','거래유형_중개거래':'trans_type_indirect'
                                                                ,'거래유형_직거래':'trans_type_direct','전용면적':'area','건물연식':'build_age','거래금액(만원)':'price'                                                      
                                                                })
            if 'trans_type_null' not in transaction_data.columns:
                transaction_data['trans_type_null'] = 0
            transaction_data = transaction_data[(transaction_data['price']>=10000)&(transaction_data['price']<=500000)]
            #3.1.3 Load dict_postal_data
            with open(os.path.join(data_dir, "dict_postal_data.pkl"), "rb")  as file:
                self.dict_postal_data = pickle.load(file)
                
            postal_node_adjacency = pd.read_csv(os.path.join(data_dir, 'postal_node_adjacency.csv'), sep=',')
            postal_node_adjacency = postal_node_adjacency.rename(columns={'우편번호':'postal_code'})
            postal_node_adjacency['postal_code'] = postal_node_adjacency['postal_code'].apply(lambda x: str(x).zfill(5))
            transaction_data = transaction_data[transaction_data['postal_code'].isin(postal_node_adjacency['postal_code'])]
            postal_node_adjacency = postal_node_adjacency.set_index(keys='postal_code')
            node_id_to_index = {node_id: idx for idx, node_id in enumerate(postal_node_adjacency.index)}
            # Prepare edge list for the graph
            edge_index = []
            for i, row in postal_node_adjacency.iterrows():
                neighbors = row[row == 1].index.values
                for neighbor in neighbors:
                    # Map the node ID and neighbor ID to integer indices
                    edge_index.append([node_id_to_index[i], node_id_to_index[neighbor]])

            # Convert the edge list to a tensor
            self.postal_code_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            #3.1.3 Load dict_addr_data
            """
            with open(os.path.join(data_dir, "dict_road_addr_data.pkl"), "rb")  as file:
                dict_road_addr_data = pickle.load(file)
            road_addr_node_adjacency = pd.read_csv(os.path.join(data_dir, 'road_addr_node_adjacency.csv'), sep=',')
            """
            #3.2 Train:Valid:Test spilit 70:15:15
            X = transaction_data[['trans_date','year_month', 'road_addr','addr','postal_code','build_num','floor',
                                'trans_type_null','trans_type_indirect','trans_type_direct','area','build_age',
                                'interest','maturity','usdkrw','kospi',
                                'tube_cnt_1km','busstop_cnt_1km','util_cnt_2km','amenity_cnt_2km','park_cnt_2km','util_cnt_2km','reconstruct_2km',
                                'school_cnt_5km','childcare_cnt_5km']]
            
            #y = np.log10(transaction_data['price'])
            y = transaction_data['price']/1000
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=pid)
            X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=pid)

            print(f"X_all: {X.shape}, y_all: {y.shape}")
            
            # Prepare datasets
            print('Setup Train Dataset')
            print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            train_dataset = self.prepare_temporal_dataset(X_train, y_train)
            print('Setup Valid Dataset')
            print(f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
            val_dataset = self.prepare_temporal_dataset(X_valid, y_valid)
            print('Setup Test Dataset')
            print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            test_dataset = self.prepare_temporal_dataset(X_test,y_test)
            with open(train_path, 'wb') as f:
                pickle.dump(train_dataset, f)
                
            with open(val_path, 'wb') as f:
                pickle.dump(val_dataset, f)
                
            with open(test_path, 'wb') as f:
                pickle.dump(test_dataset, f)

        self.batch_size = self.config.batch_size
        # Prepare DataLoader for batch processing
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        #set_algorithm
        for alg in algs:
            alg.set_algo(model_dir,start_epoch)



    def prepare_dataset(self,data_x,data_y):
        postal_codes = data_x['postal_code'].values
        year_months = data_x['year_month'].values
        transaction_amounts = data_y.values
        apart_features =  data_x[['floor','trans_type_null','trans_type_indirect','trans_type_direct','area','build_age','interest','maturity','usdkrw','kospi']].values
        apart_poi = data_x[['tube_cnt_1km','busstop_cnt_1km','util_cnt_2km','amenity_cnt_2km','park_cnt_2km','reconstruct_2km','school_cnt_5km','childcare_cnt_5km']].values
        dataset = []
        for i in tqdm(range(len(data_x)), total=len(data_x), leave=True):
            year_month = year_months[i]
            postal_code = postal_codes[i]
            transaction_amount = transaction_amounts[i]
            
            features = self.dict_postal_data[year_month].drop(columns=['postal_code']).values
            features = torch.tensor(features, dtype=torch.float)
            
            data_item = Data(
                x=features, 
                edge_index=self.postal_code_edge_index, 
                y=torch.tensor([transaction_amount], dtype=torch.float)
            )
            data_item.node_idx = self.dict_postal_data[year_month].index[self.dict_postal_data[year_month]['postal_code'] == postal_code][0]
            data_item.apart_feature  = torch.tensor(apart_features[i], dtype=torch.float)
            num_nodes = len(self.dict_postal_data[year_month])
            one_hot_temp = torch.zeros(num_nodes,dtype=torch.float)
            one_hot_temp[data_item.node_idx] = 1
            data_item.trans_poi  = torch.cat([data_item.apart_feature ,torch.tensor(apart_poi[i], dtype=torch.float), one_hot_temp], dim=0)
            dataset.append(data_item)

        return dataset
    
    def prepare_temporal_dataset(self,data_x,data_y):
        postal_codes = data_x['postal_code'].values
        year_months = data_x['year_month'].values
        transaction_amounts = data_y.values
        apart_features =  data_x[['floor','trans_type_null','trans_type_indirect','trans_type_direct','area','build_age','interest','maturity','usdkrw','kospi']].values
        apart_poi = data_x[['tube_cnt_1km','busstop_cnt_1km','util_cnt_2km','amenity_cnt_2km','park_cnt_2km','reconstruct_2km','school_cnt_5km','childcare_cnt_5km']].values
        dataset = []
        for i in tqdm(range(len(data_x)), total=len(data_x), leave=True):
            postal_code = postal_codes[i]
            transaction_amount = transaction_amounts[i]

            year_month = year_months[i]        
            year = year_month // 100
            month = year_month % 100
            intervals = np.zeros(11,dtype = int)
            for i in range(11):  # 현재부터 5년 전까지 (6개월 간격, 11개 구간)
                offset_year = year - (i // 2)
                offset_month = month - (6 * (i % 2))
                if offset_month <= 0:
                    offset_year -= 1
                    offset_month += 12
                intervals[10 - i] = offset_year*100+offset_month
            temporal_data_item = []
            for ym in intervals:
                features = self.dict_postal_data[ym].drop(columns=['postal_code']).values
                features = torch.tensor(features, dtype=torch.float)
                data_item = Data(
                    x=features, 
                    edge_index=self.postal_code_edge_index, 
                    y=torch.tensor([transaction_amount], dtype=torch.float)
                )
                data_item.node_idx = self.dict_postal_data[year_month].index[self.dict_postal_data[year_month]['postal_code'] == postal_code][0]
                data_item.apart_feature  = torch.tensor(apart_features[i], dtype=torch.float)
                num_nodes = len(self.dict_postal_data[year_month])
                one_hot_temp = torch.zeros(num_nodes,dtype=torch.float)
                one_hot_temp[data_item.node_idx] = 1
                data_item.trans_poi  = torch.cat([data_item.apart_feature ,torch.tensor(apart_poi[i], dtype=torch.float), one_hot_temp], dim=0)
                temporal_data_item.append(data_item)
            dataset.append(temporal_data_item)

        return dataset
    
    def train(self, algs, pid, model_dir):
        start_time = datetime.datetime.now()
        start_epoch = self.config.start_epoch
        num_epochs = self.config.num_epochs
        algo_names = [alg.name for alg in algs]
        print(f'Algos: {algo_names}')

        if self.config.temporal:
            self.prepare_temporal(algs, int(pid), model_dir,start_epoch)
        else:    
            self.prepare(algs, int(pid), model_dir,start_epoch)

        min_loss_dict ={alg_name: float('inf') for alg_name in algo_names}
        if start_epoch != 0:
            for alg in algs:
                for i in range(math.floor(start_epoch/10)):
                    alg.scheduler.step()
                    min_loss_epoch = min(alg.val_loss_dict, key=lambda epoch: np.mean(alg.val_loss_dict[epoch]))
                    min_loss_dict[alg.name] = np.mean(alg.val_loss_dict[min_loss_epoch])
        for epoch in range(start_epoch, num_epochs):
            seed = pid * 10000 + epoch
            set_seed(seed)
            for alg in algs:
                total_train_loss = 0
                total_val_loss = 0
                train_losses = []
                val_losses = []
                val_predictions = []
                val_targets = []
                alg.train()
                for batch in self.train_loader:
                    if self.config.temporal:
                        y = batch[0].y.to(self.device)
                    else:
                        batch = batch.to(self.device)
                        y = batch.y
                    alg.optimizer.zero_grad()
                    output = alg(batch)
                    loss = alg.criterion(output.view(-1), y)
                    loss.backward()
                    alg.optimizer.step()
                    train_losses.append(loss.item())
                    total_train_loss += loss.item() * self.batch_size
                if (epoch+1)%10 == 0:
                    alg.scheduler.step()
                avg_train_loss = total_train_loss / len(self.train_loader.dataset)
                alg.train_loss_dict[epoch] = train_losses
                
                alg.eval()
                with torch.no_grad():
                    for batch in self.val_loader:
                        if self.config.temporal:
                            y = batch[0].y.to(self.device)
                        else:
                            batch = batch.to(self.device)
                            y = batch.y
                        alg.optimizer.zero_grad()
                        output = alg(batch)
                        loss = alg.criterion(output.view(-1),y)
                        total_val_loss += loss.item() * self.batch_size
                        val_losses.append(loss.item())
                        val_predictions.append(output.view(-1))
                        val_targets.append(y)
                val_targets = torch.cat(val_targets)
                val_predictions = torch.cat(val_predictions)

                correlation_matrix_pred = torch.corrcoef(torch.stack((val_targets, val_predictions)))
                correlation_pred = correlation_matrix_pred[0, 1].item()

                avg_val_loss = total_val_loss / len(self.val_loader.dataset)

                alg.val_loss_dict[epoch] = val_losses
                alg.correlation_preds.append(correlation_pred)
                
                print(f'Epoch {epoch + 1}/{num_epochs}, Algo: {alg.name} Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_val_loss:.2f}, Validation Coeff: {correlation_pred:.4f}')
                if min_loss_dict[alg.name] > avg_val_loss:
                    min_loss_dict[alg.name] =avg_val_loss
                    print(f'Best model update (epoch = {epoch+1})')
                    os.makedirs(f"results/{model_dir}/{alg.name}/", exist_ok=True)
                    torch.save(alg.model.state_dict(), f"models/{model_dir}/{alg.name}/best_model.pth")
                if ((epoch + 1) % 5 == 0) or (epoch+1 == num_epochs):
                    print(f'현재 진행 시간: {(datetime.datetime.now() - start_time).seconds} sec')
                    torch.save(alg.model.state_dict(), f"models/{model_dir}/{alg.name}/{epoch+1}.pth")
                    with open(f"results/{model_dir}/{alg.name}/train_loss.json", 'w') as file:
                        json.dump(alg.train_loss_dict, file)
                    with open(f"results/{model_dir}/{alg.name}/val_loss.json", 'w') as file:
                        json.dump(alg.val_loss_dict, file)
                    with open(f"results/{model_dir}/{alg.name}/val_corr.json", 'w') as file:
                        json.dump(alg.correlation_preds, file)


    def eval(self,algs,model_dir):
        algo_names = [alg.name for alg in algs]
        print(f'Algos: {algo_names}')
        if self.config.temporal:
            self.prepare_temporal(algs, int(pid), model_dir,None)
            model_dir = model_dir+'_temporal'
        else:    
            self.prepare(algs, int(pid), model_dir,None)
        results = {
            'Algorithm': [],
            'MSE': [],
            'RMSE':[], 
            'MAPE': [],
            'Pearson Coefficient': []
        }
        for alg in algs:
            test_predictions = []
            test_targets = []
            alg.eval()
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc=f"Test {alg.name}:", total=len(self.test_loader.dataset)//self.batch_size):
                    if self.config.temporal:
                        y = batch[0].y.to(self.device)
                    else:
                        batch = batch.to(self.device)
                        y = batch.y
                    output = alg(batch)
                    test_predictions.append(output.view(-1))
                    test_targets.append(y)
            test_targets = torch.cat(test_targets)
            test_predictions = torch.cat(test_predictions)
            
            mse = torch.mean((test_predictions - test_targets) ** 2).item()
            rmse = torch.mean(torch.sqrt((test_predictions - test_targets) ** 2)).item()
            mape = torch.mean(torch.abs((test_predictions - test_targets) / test_targets)).item()
            correlation_matrix_pred = torch.corrcoef(torch.stack((test_targets, test_predictions)))
            pearson_coeff = correlation_matrix_pred[0, 1].item()


            results['Algorithm'].append(alg.name)
            results['MSE'].append(mse)
            results['RMSE'].append(rmse)
            results['MAPE'].append(mape)
            results['Pearson Coefficient'].append(pearson_coeff)
            os.makedirs(f"results/{model_dir}", exist_ok=True)
            plt.figure(figsize=(8, 6))
            plt.scatter(test_targets.cpu().numpy(), test_predictions.cpu().numpy(), alpha=0.7)
            min_val = min(np.min(test_targets.cpu().numpy()), np.min(test_predictions.cpu().numpy()))
            max_val = max(np.max(test_targets.cpu().numpy()), np.max(test_predictions.cpu().numpy()))
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction (y=x)')
            plt.title(f'{alg.name}: Scatterplot of Predictions vs Targets', fontsize=14)
            plt.xlabel('True Values', fontsize=12)
            plt.ylabel('Predicted Values', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"results/{model_dir}/scatter_{alg.name}.png")
            plt.close()
            print(f'Algo: {alg.name} MSE: {mse:.4f},RMSE: {rmse:.2f}, MAPE: {mape:.4f}, Pearson Coefficient: {pearson_coeff:.4f}')
        

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"results/{model_dir}.csv", index=False)
        
        metrics = ['MSE','RMSE', 'MAPE', 'Pearson Coefficient']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.bar(results['Algorithm'], results[metric])
            plt.title(f'{metric} by Algorithm')
            plt.xlabel('Algorithm')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"results/{model_dir}/{metric}.png")
            plt.close()


if __name__ =="__main__":
    #1. load configuration file (utill.py: Config & find_option)
    device = find_option("cuda","0")
    config_name = find_option("config")
    if config_name.endswith(".config"):
        config_name = config_name[:-len(".config")]
    config = Config('configs/'+config_name+'.config')

    #2. Setup algorithms
    algs = []
    for cat, cmd in config.algorithms:
        exec("import "+cmd[:cmd.find(".")])
        module = cmd[:-1]+", config, device)"
        alg = eval(module)
        algs.append(alg)

    if len(algs) == 0:
        raise Exception( "no algorithm is configured" )
    device = int(device)
    #3. Setup data
    pid = find_option("pid","0")
    train_eval = Train_Eval(config,device)
    print(f'PID = {pid}')
    #4. Training or Evaluation
    model_dir = pid+'_'+str(config.years[0])+'_'+str(config.years[-1])
    if "train" in config_name:
        if model_dir not in os.listdir("models"):
            os.mkdir("models/"+model_dir)
        for alg in algs:
            if alg.name not in os.listdir("models/"+model_dir):
                os.mkdir("models/"+model_dir+"/"+alg.name)
        train_eval.train(algs,int(pid),model_dir)
    else:
        train_eval.eval(algs,model_dir)