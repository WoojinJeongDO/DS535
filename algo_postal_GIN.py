import json
import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout,LeakyReLU
from torch_geometric.nn import GINConv
from algo_common import *
from utill import *



class Alg_Postal_GIN(Alg):
    def set_algo(self,model_dir,start_epoch):
        self.model = Postal_GIN(12,10,64,128,64).to(self.device)

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



class Postal_GIN(torch.nn.Module):
    def __init__(self, input_dim, apart_feature_dim, hidden_dim, hidden_dim2, hidden_dim3):
        super(Postal_GIN, self).__init__()
        # GINConv with adjusted MLP
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
        
        # Fully connected layers
        self.fc1 = Linear(hidden_dim + apart_feature_dim, hidden_dim2)
        self.fc2 = Linear(hidden_dim2, hidden_dim3)
        self.fc3 = Linear(hidden_dim3, 1)

        # Activation functions and other modules
        self.act1 = LeakyReLU(negative_slope=0.1)
        self.act2 = LeakyReLU(negative_slope=0.05)
        self.sigmoid = torch.nn.Sigmoid()  # 필요에 따라 사용

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index 
        target_node_idx = batch.node_idx
        apart_feature = batch.apart_feature 
        
        # Apply GINConv layers
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        
        # Combine GNN features with apart features
        combined_features = torch.cat((x[target_node_idx], apart_feature.view(-1, 10)), dim=1)
        
        # Fully connected layers
        x = self.fc1(combined_features)
        x = self.act1(x)
        
        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x)
        
        return x