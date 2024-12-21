import json
import torch
from torch_geometric.nn import SAGEConv
from algo_common import *
from utill import *



class Alg_Postal_GraphSage(Alg):
    def set_algo(self,model_dir,start_epoch):
        self.model = Postal_GraphSage(12,10,64,128,64).to(self.device)

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



class Postal_GraphSage(torch.nn.Module):
    def __init__(self, input_dim, apart_feature_dim, hidden_dim, hidden_dim2, hidden_dim3):
        super(Postal_GraphSage, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        self.fc1 = torch.nn.Linear(hidden_dim+apart_feature_dim, hidden_dim2)
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
        target_node_idx = batch.node_idx
        apart_feature = batch.apart_feature 
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc1( torch.cat((x[target_node_idx], apart_feature.view(-1, 10)), dim=1))
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x)
        return x