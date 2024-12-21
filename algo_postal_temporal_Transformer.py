import json
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from algo_common import *
from utill import *
from torch.utils.checkpoint import checkpoint
import math

class Alg_postal_temporal_Transformer(Alg):
    def set_algo(self,model_dir,start_epoch):
        # embed_dim,apart_feature_dim,hidden_dim2,dim_feedforward, dropout, num_layers
        self.model = Postal_temporal_Transformer(8,10,32,32,0.1,1)

        if self.training_mode:
            self.train_loss_dict = {}
            self.val_loss_dict = {}
            self.correlation_preds = []
            if start_epoch>0:
                model_path = f"models/{model_dir}/{self.name}/{start_epoch}.pth"
                if os.path.exists(model_path):
                    print(f"Loading model from {model_path}")
                    self.model.load_state_dict(torch.load(model_path, weights_only=True))
                    # self.model.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=True))
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
                state_dict = torch.load(f"models/{model_dir}/{self.name}/best_model.pth")
                from collections import OrderedDict
    # Create a new OrderedDict to hold the updated state_dict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        # Remove the 'module.' prefix
                        new_key = k[7:]
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(torch.load(f"models/{model_dir}/{self.name}/{self.config.epoch}.pth"))
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
            self.criterion = torch.nn.MSELoss()
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)
        
torch.autograd.set_detect_anomaly(True)

class Postal_temporal_Transformer(torch.nn.Module):
    def __init__(self, embed_dim,apart_feature_dim,hidden_dim2,dim_feedforward, dropout, num_layers):
        super(Postal_temporal_Transformer, self).__init__()

        self.embedding = nn.Linear(1, embed_dim)
        num_heads=2
        # Transformer 인코더 레이어
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        
        self.dropout = nn.Dropout(dropout)

        self.fc1 = torch.nn.Linear(embed_dim+apart_feature_dim, hidden_dim2)
        self.fc2 = torch.nn.Linear(hidden_dim2, 1)
        self.act1 = torch.nn.LeakyReLU(negative_slope=0.1)
        

    def forward(self, batch, src_key_padding_mask=None):
        seqs, attention_masks, apart_features = batch
        assert not torch.isnan(seqs).any(), "NaNs found in seqs"
        assert not torch.isinf(seqs).any(), "Infs found in seqs"
        assert not torch.isnan(attention_masks).any(), "NaNs found in attention_masks"
        assert not torch.isinf(attention_masks).any(), "Infs found in attention_masks"
        assert not torch.isnan(apart_features).any(), "NaNs found in apart_features"
        assert not torch.isinf(apart_features).any(), "Infs found in apart_features"

        if attention_masks.dtype != torch.bool:
            attention_masks = attention_masks.bool()
        seqs = self.embedding(seqs.unsqueeze(-1))
        seqs = seqs.permute(1, 0, 2)
        assert not torch.isnan(seqs).any(), "NaNs found in embedding output"
        assert not torch.isinf(seqs).any(), "Infs found in embedding output"

        transformer_output = self.transformer_encoder(seqs)  # [seq_length, batch_size, embed_dim]
        assert not torch.isnan(transformer_output).any(), "NaNs found after transformer encoder"

        # print(transformer_output)
        # 출력 재배치 및 집약 (예: 평균 풀링)
        transformer_output = transformer_output.permute(1, 0, 2)  # [batch_size, seq_length, embed_dim]
        pooled_output = transformer_output.mean(dim=1)  # [batch_size, embed_dim]
        
        # 최종 출력
        tf_output = self.dropout(pooled_output)

        # apart_feature: [10 * batch_size] -> [batch_size, 10]
        
        # (num_time_slot, batch_size, 10)로 broadcast
        # apart_feature_expand = apart_feature.unsqueeze(0).expand(self.num_time_slot, batch_size, 10)
        
        # concat: [num_time_slot, batch_size, hidden_dim+10]
        region_embeddings_att3 = torch.cat((tf_output,apart_features), dim=1)
        x = self.fc1(region_embeddings_att3)
        x = self.act1(x)
        x = self.fc2(x)
        return x
    


    

