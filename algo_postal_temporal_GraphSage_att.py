import json
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from algo_common import *
from utill import *



class Alg_postal_temporal_GraphSage_att(Alg):
    def set_algo(self,model_dir,start_epoch):
        self.model = Postal_temporal_GraphSage_att(12,10,64,128,64,11,5611,self.device).to(self.device)

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



class Postal_temporal_GraphSage_att(torch.nn.Module):
    def __init__(self, input_dim, apart_feature_dim, hidden_dim, hidden_dim2, hidden_dim3,num_time_slot,num_postal_codes,device):
        super(Postal_temporal_GraphSage_att, self).__init__()
        self.device = device
        self.num_time_slot = num_time_slot
        self.num_postal_codes = num_postal_codes
        self.hidden_dim = hidden_dim
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.attention = ScaledDotProductAttention(hidden_dim)

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

        # Initialize region_embeddings
        region_embeddings = []

        # First convolutional layer + activation
        for i in range(self.num_time_slot):
            x = batch[i].to(self.device).x.view(-1, 12)
            x = self.conv1(x, edge_index)
            region_embeddings.append(torch.relu(x))

        # Convert list to tensor
        region_embeddings = torch.stack(region_embeddings, dim=0)  # Shape: (num_time_slot, num_nodes, hidden_dim)

        # Apply attention mechanism
        region_embeddings = self.attention(region_embeddings.permute(1, 0, 2)).permute(1, 0, 2)

        # Second convolutional layer + activation
        temp_region_embeddings = []
        for i in range(self.num_time_slot):
            x = region_embeddings[i]
            x = self.conv2(x, edge_index)
            temp_region_embeddings.append(torch.relu(x))
        region_embeddings = torch.stack(temp_region_embeddings, dim=0)

        # Apply attention mechanism and reduce to a single representation per node
        region_embeddings = self.attention(region_embeddings.permute(1, 0, 2)).sum(dim=1)  # Shape: (num_nodes, hidden_dim)

        # Fully connected layers
        x = torch.cat((region_embeddings[target_node_idx], apart_feature.view(-1, 10)), dim=1)
        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x)
        return x

    



class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # Query, Key, Value를 위한 Linear Layer 정의
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x의 shape: (batch_size, time_steps, feature_dim)
        queries = self.query_linear(x) / (self.hidden_dim ** 0.5)  # Scaled Query 계산
        keys = self.key_linear(x)                                 # Key 계산
        values = self.value_linear(x)                             # Value 계산

        # Attention Score 계산: (batch_size, time_steps, time_steps)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = F.softmax(attention_scores, dim=-1)   # Softmax로 정규화

        # Attention Output 계산: (batch_size, time_steps, feature_dim)
        attention_output = torch.matmul(attention_weights, values)

        # Time Steps 차원(sum dim=1)을 없애고 최종 출력 반환
        return attention_output  # (batch_size, feature_dim)