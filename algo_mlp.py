import json
import torch
from algo_common import *
from utill import *



class Alg_MLP(Alg):
    def set_algo(self,model_dir,start_epoch):
        self.model = MLP(5624,64,64).to(self.device) #input:postal_code one hot 5611 + apart_features 6 + apart_poi 7 = 5624

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



class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.act1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.act2 = torch.nn.LeakyReLU(negative_slope=0.05)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size1)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size2)

    def forward(self, batch):
        batch_size = len(batch.ptr) - 1
        x = batch.trans_poi.view(batch_size, -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.fc3(x)
        return x
