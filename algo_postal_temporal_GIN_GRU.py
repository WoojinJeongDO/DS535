import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import GRU
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout,LeakyReLU
from torch_geometric.nn import GINConv
from algo_common import *
from utill import *



class Alg_postal_temporal_GIN_GRU(Alg):
    def set_algo(self,model_dir,start_epoch):
        self.model = Postal_temporal_GIN_GRU(12,10,64,128,64,11,5611,self.device).to(self.device)

        if self.training_mode:
            self.train_loss_dict = {}
            self.val_loss_dict = {}
            self.correlation_preds = []
            if start_epoch>0:
                model_path = f"models/{model_dir}/{self.name}/{start_epoch}.pth"
                if os.path.exists(model_path):
                    print(f"Loading model from {model_path}")
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=True))
                    with open(f"results/{model_dir}/{self.name}/train_loss.json", 'r') as file:
                        self.train_loss_dict = json.load(file)
                    with open(f"results/{model_dir}/{self.name}/val_loss.json", 'r') as file:
                        self.val_loss_dict = json.load(file)
                    with open(f"results/{model_dir}/{self.name}/val_corr.json", 'r') as file:
                        self.correlation_preds = json.load(file)
                else:
                    error_msg = f"No model found at {model_path}. Starting from scratch for {self.name}."
                    print(error_msg)
                    raise FileNotFoundError(error_msg)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
            self.criterion = torch.nn.MSELoss()
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)
        else:
            if self.best_model:
                self.model.load_state_dict(torch.load(f"models/{model_dir}/{self.name}/best_model.pth", map_location=self.device, weights_only=True))
            else:
                self.model.load_state_dict(torch.load(f"models/{model_dir}/{self.name}/{self.config.epoch}.pth", map_location=self.device, weights_only=True))



class Postal_temporal_GIN_GRU(torch.nn.Module):
    def __init__(self, input_dim, apart_feature_dim, hidden_dim, hidden_dim2, hidden_dim3,num_time_slot,num_postal_codes,device):
        super(Postal_temporal_GIN_GRU, self).__init__()
        self.device = device
        self.num_time_slot = num_time_slot
        self.num_postal_codes = num_postal_codes
        self.hidden_dim = hidden_dim
        self.conv1 = GINConv(
            Sequential(
                Linear(input_dim, hidden_dim),
                ReLU(),
                BatchNorm1d(hidden_dim),
                Dropout(p=0.1),  # Dropout 추가
                Linear(hidden_dim, hidden_dim),
                ReLU()
            )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BatchNorm1d(hidden_dim),
                Dropout(p=0.1),
                Linear(hidden_dim, hidden_dim),
                ReLU()
            )
        )
        self.GRU_layer = GRU(input_size=hidden_dim,hidden_size=hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim+apart_feature_dim, hidden_dim2)
        self.fc2 = torch.nn.Linear(hidden_dim2, hidden_dim3)
        self.fc3 = torch.nn.Linear(hidden_dim3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.act1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.act2 = torch.nn.LeakyReLU(negative_slope=0.05)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim3)

    def forward(self, batch):
        batch_0 = batch[0].to(self.device)
        edge_index = batch_0.edge_index.view(2, -1)
        target_node_idx = batch_0.node_idx
        apart_feature = batch_0.apart_feature
        

        # Time slot 별 region_embeddings 계산
        temp_region_embeddings = []
        for i in range(self.num_time_slot):
            x = batch[i].to(self.device).x.view(-1, 12)
            x = self.conv1(x, edge_index)
            temp_region_embeddings.append(torch.relu(x))
        region_embeddings = torch.stack(temp_region_embeddings, dim=0)

        region_embeddings, _ = self.GRU_layer(region_embeddings)
        
        # 두 번째 GINConv 연산
        temp_region_embeddings = []
        for i in range(self.num_time_slot):    
            x = region_embeddings[i]
            x = self.conv2(x, edge_index)
            temp_region_embeddings.append(torch.relu(x))
        region_embeddings = torch.stack(temp_region_embeddings, dim=0)

        # GRU 연산 후 region_embeddings 업데이트
        _, region_embeddings = self.GRU_layer(region_embeddings)
        region_embeddings = region_embeddings.squeeze(0)

        # Fully connected layers
        x = self.fc1(torch.cat((region_embeddings[target_node_idx], apart_feature.view(-1, 10)), dim=1))
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x)
        return x