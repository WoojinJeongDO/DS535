import json
import torch
from torch_geometric.nn import SAGEConv
from algo_common import *
from utill import *



class Alg_trans_GNN(Alg):
    def set_algo(self,model_dir,start_epoch):
        self.model = trans_GNN(16,128,256,128).to(self.device)

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
        else:
            self.model.load_state_dict(torch.load(f"models/{model_dir}/{self.name}/{self.config.epoch}.pth", map_location=self.device))



class trans_GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, hidden_dim3):
        super(trans_GNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = torch.nn.Linear(hidden_dim2, hidden_dim3)
        self.fc3 = torch.nn.Linear(hidden_dim3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.act1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.act2 = torch.nn.LeakyReLU(negative_slope=0.05)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim3)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index 
        target_node_idx = batch.node_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc1( x[target_node_idx] )
        x = self.bn1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.fc3(x)
        return x