import datetime
import os
import math
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from algo_common import *
from utill import *

# accelerate 임포트
from accelerate import Accelerator

import torch
import torch.optim as optim  # 옵티마이저 임포트 추가

class Train_Eval:
    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator  # accelerator 사용
        self.device = accelerator.device  # 이전 코드 호환을 위해 device 설정

    def prepare_tf(self, algs, pid, model_dir, start_epoch):
        from torch.utils.data import Dataset

        class PandasDataset(Dataset):
            def __init__(self, dataframe):
                self.df = dataframe

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                seq = np.concatenate(row['seq'])
                apart_features = row['apart_features']
                label = row['y']

                return (
                    torch.tensor(seq, dtype=torch.float32),
                    torch.tensor(apart_features, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.float32),
                )

        set_seed(pid)
        # 데이터 로딩
        data_dir = 'data/'
        data_path = f'data/{model_dir}'
        # train_path = f'data/{model_dir}/test_temporal_transformer_dataset_temp.pkl'
        # val_path = f'data/{model_dir}/test_temporal_transformer_dataset_temp.pkl'
        # test_path = f'data/{model_dir}/test_temporal_transformer_dataset_temp.pkl'
        train_path = f'data/{model_dir}/train_temporal_transformer_dataset.pkl'
        val_path = f'data/{model_dir}/val_temporal_transformer_dataset.pkl'
        test_path = f'data/{model_dir}/test_temporal_transformer_dataset.pkl'

        if dataset_exists(train_path) & dataset_exists(val_path) & dataset_exists(test_path):
            # 저장된 파일에서 데이터셋 로드
            print("Loading datasets from saved files...")
            with open(train_path, 'rb') as f:
                train_loaded = pickle.load(f)
                train_dataset = PandasDataset(train_loaded)

            with open(val_path, 'rb') as f:
                val_loaded = pickle.load(f)
                val_dataset = PandasDataset(val_loaded)

            with open(test_path, 'rb') as f:
                test_loaded = pickle.load(f)
                test_dataset = PandasDataset(test_loaded)

            self.batch_size = self.config.batch_size

            from torch.nn.utils.rnn import pad_sequence

            def collate_fn(batch):
                """
                batch: list of tuples (seq, apart_features, y)
                """
                seqs, apart_features, ys = zip(*batch)

                # 시퀀스 패딩
                padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)  # [batch_size, max_seq_len]

                # 어텐션 마스크 생성 (패딩된 부분은 0, 실제 데이터는 1)
                attention_masks = torch.zeros(padded_seqs.shape, dtype=torch.bool)
                for i, seq in enumerate(seqs):
                    attention_masks[i, :len(seq)] = 1

                # 나머지 피처 스택
                apart_features = torch.stack(apart_features)  # [batch_size, 10]
                ys = torch.stack(ys)  # [batch_size]

                return padded_seqs, attention_masks, apart_features, ys

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

            # 알고리즘 설정
            for alg in algs:
                alg.set_algo(model_dir, start_epoch)
            # print(algs)
            # print(algs[0])
            # accelerator를 사용하여 데이터로더와 모델 준비
            model, optimizer, train_loader, val_loader, test_loader, algs = self.accelerator.prepare(algs[0].model, algs[0].optimizer,
                train_loader, val_loader, test_loader, *algs
            )
            algs.model = model
            algs.optimizer = optimizer
            # 나중에 사용하기 위해 할당
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

        else:
            raise RuntimeError("No preprocess tf dataset")

    def train(self, algs, pid, model_dir):
        start_time = datetime.datetime.now()
        start_epoch = self.config.start_epoch
        num_epochs = self.config.num_epochs
        algo_names = [alg.name for alg in algs]
        print(f'Algos: {algo_names}')

        if self.config.temporal:
            self.prepare_temporal(algs, int(pid), model_dir, start_epoch)
        elif self.config.tf:
            self.prepare_tf(algs, int(pid), model_dir, start_epoch)
        else:
            self.prepare(algs, int(pid), model_dir, start_epoch)

        min_loss_dict = {alg_name: float('inf') for alg_name in algo_names}
        if start_epoch != 0:
            for alg in algs:
                for i in range(math.floor(start_epoch / 10)):
                    alg.scheduler.step()
                min_loss_epoch = min(
                    alg.val_loss_dict,
                    key=lambda epoch: np.mean(alg.val_loss_dict[epoch]),
                )
                min_loss_dict[alg.name] = np.mean(alg.val_loss_dict[min_loss_epoch])

        # 에포크 반복
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Training Epochs"):
            seed = pid * 10000 + epoch
            set_seed(seed)
            for alg in algs:
                total_train_loss = 0
                total_val_loss = 0
                train_losses = []
                val_losses = []
                val_predictions = []
                val_targets = []
                alg.model.train()  # 모델을 학습 모드로 설정

                # 학습 루프
                for batch in tqdm(self.train_loader, desc=f"Training {alg.name}"):
                    seqs, attention_masks, apart_features, ys = batch
                    # 텐서를 올바른 장치로 이동
                    # seqs = seqs.to(self.accelerator.device)
                    # attention_masks = attention_masks.to(self.accelerator.device)
                    # apart_features = apart_features.to(self.accelerator.device)
                    # ys = ys.to(self.accelerator.device)

                    alg.optimizer.zero_grad()
                    output = alg((seqs, attention_masks, apart_features))
                    loss = alg.criterion(output.view(-1), ys)
                    # 역전파를 accelerator를 통해 수행
                    self.accelerator.backward(loss)
                    alg.optimizer.step()

                    train_losses.append(loss.item())
                    total_train_loss += loss.item() * self.batch_size

                if (epoch + 1) % 10 == 0:
                    alg.scheduler.step()

                avg_train_loss = total_train_loss / len(self.train_loader.dataset)
                alg.train_loss_dict[epoch] = train_losses

                # 검증 루프
                alg.model.eval()
                with torch.no_grad():
                    for batch in tqdm(self.val_loader, desc=f"Validating {alg.name}"):
                        seqs, attention_masks, apart_features, ys = batch
                        # 텐서를 올바른 장치로 이동
                        # seqs = seqs.to(self.accelerator.device)
                        # attention_masks = attention_masks.to(self.accelerator.device)
                        # apart_features = apart_features.to(self.accelerator.device)
                        # ys = ys.to(self.accelerator.device)

                        output = alg((seqs, attention_masks, apart_features))
                        loss = alg.criterion(output.view(-1), ys)

                        total_val_loss += loss.item() * self.batch_size
                        val_losses.append(loss.item())
                        val_predictions.append(output.view(-1))
                        val_targets.append(ys)

                val_targets = torch.cat(val_targets)
                val_predictions = torch.cat(val_predictions)
                correlation_matrix_pred = torch.corrcoef(
                    torch.stack((val_targets, val_predictions))
                )
                correlation_pred = correlation_matrix_pred[0, 1].item()

                avg_val_loss = total_val_loss / len(self.val_loader.dataset)

                alg.val_loss_dict[epoch] = val_losses
                alg.correlation_preds.append(correlation_pred)

                if self.accelerator.is_main_process:
                    print(
                        f'Epoch {epoch + 1}/{num_epochs}, Algo: {alg.name} '
                        f'Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_val_loss:.2f}, '
                        f'Validation Coeff: {correlation_pred:.4f}'
                    )

                if min_loss_dict[alg.name] > avg_val_loss:
                    min_loss_dict[alg.name] = avg_val_loss
                    if self.accelerator.is_main_process:
                        print(f'Best model update (epoch = {epoch +1})')
                        os.makedirs(
                            f"results/{model_dir}/{alg.name}/", exist_ok=True
                        )
                        torch.save(
                            alg.model.state_dict(),
                            f"models/{model_dir}/{alg.name}/best_model_{epoch}.pth",
                        )

                if ((epoch + 1) % 5 == 0) or (epoch + 1 == num_epochs):
                    if self.accelerator.is_main_process:
                        print(
                            f'현재 진행 시간: {(datetime.datetime.now() - start_time).seconds} sec'
                        )
                        torch.save(
                            alg.model.state_dict(),
                            f"models/{model_dir}/{alg.name}/{epoch +1}.pth",
                        )
                        with open(
                            f"results/{model_dir}/{alg.name}/train_loss.json", "w"
                        ) as file:
                            json.dump(alg.train_loss_dict, file)
                        with open(
                            f"results/{model_dir}/{alg.name}/val_loss.json", "w"
                        ) as file:
                            json.dump(alg.val_loss_dict, file)
                        with open(
                            f"results/{model_dir}/{alg.name}/val_corr.json", "w"
                        ) as file:
                            json.dump(alg.correlation_preds, file)

    def eval(self, algs, model_dir):
        algo_names = [alg.name for alg in algs]
        print(f'Algos: {algo_names}')
        if self.config.temporal:
            self.prepare_temporal(algs, int(pid), model_dir, None)
        elif self.config.tf:
            self.prepare_tf(algs, int(pid), model_dir, None)
        else:
            self.prepare(algs, int(pid), model_dir, None)

        results = {
            'Algorithm': [],
            'MSE': [],
            'RMSE': [],
            'MAPE': [],
            'Pearson Coefficient': [],
        }

        for alg in algs:
            test_predictions = []
            test_targets = []
            alg.model.eval()
            with torch.no_grad():
                for batch in tqdm(
                    self.test_loader, desc=f"Test {alg.name}:"
                ):
                    seqs, attention_masks, apart_features, ys = batch
                    # 텐서를 올바른 장치로 이동
                    # seqs = seqs.to(self.accelerator.device)
                    # attention_masks = attention_masks.to(self.accelerator.device)
                    # apart_features = apart_features.to(self.accelerator.device)
                    # ys = ys.to(self.accelerator.device)

                    output = alg((seqs, attention_masks, apart_features))
                    test_predictions.append(output.view(-1))
                    test_targets.append(ys)

            test_targets = torch.cat(test_targets)
            test_predictions = torch.cat(test_predictions)

            mse = torch.mean((test_predictions - test_targets) ** 2).item()
            rmse = torch.mean(torch.sqrt((test_predictions - test_targets) ** 2)).item()
            mape = torch.mean(
                torch.abs((test_predictions - test_targets) / test_targets)
            ).item()
            correlation_matrix_pred = torch.corrcoef(
                torch.stack((test_targets, test_predictions))
            )
            pearson_coeff = correlation_matrix_pred[0, 1].item()

            results['Algorithm'].append(alg.name)
            results['MSE'].append(mse)
            results['RMSE'].append(rmse)
            results['MAPE'].append(mape)
            results['Pearson Coefficient'].append(pearson_coeff)

            if self.accelerator.is_main_process:
                plt.figure(figsize=(8, 6))
                plt.scatter(
                    test_targets.cpu().numpy(),
                    test_predictions.cpu().numpy(),
                    alpha=0.7,
                )
                plt.title(
                    f'{alg.name}: Scatterplot of Predictions vs Targets',
                    fontsize=14,
                )
                plt.xlabel('True Values', fontsize=12)
                plt.ylabel('Predicted Values', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(f"results/{model_dir}/scatter_{alg.name}.png")
                plt.close()
                print(
                    f'Algo: {alg.name} MSE: {mse:.4f}, RMSE: {rmse:.2f}, '
                    f'MAPE: {mape:.4f}, Pearson Coefficient: {pearson_coeff:.4f}'
                )

        if self.accelerator.is_main_process:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"results/{model_dir}.csv", index=False)

            metrics = ['MSE', 'RMSE', 'MAPE', 'Pearson Coefficient']
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

    def prepare(self, algs, pid, model_dir, start_epoch):
        # prepare_tf과 유사하게 구현
        pass

    def prepare_temporal(self, algs, pid, model_dir, start_epoch):
        # temporal 데이터 준비가 필요한 경우 구현
        pass

if __name__ == "__main__":
    # 알고리즘 클래스 정의 및 임포트
    import json
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from algo_common import *
    from utill import *
    from torch.utils.checkpoint import checkpoint
    import math

    # Accelerator 초기화
    accelerator = Accelerator()

    # 1. 설정 파일 로드 (utill.py: Config & find_option)
    config_name = find_option("config")
    if config_name.endswith(".config"):
        config_name = config_name[:-len(".config")]
    config = Config('configs/' + config_name + '.config')

    # 2. 알고리즘 설정
    algs = []
    for cat, cmd in config.algorithms:
        exec("import " + cmd[: cmd.find(".")])
        module = cmd[:-1] + ", config, accelerator.device)"
        alg = eval(module)
        algs.append(alg)

    if len(algs) == 0:
        raise Exception("no algorithm is configured")

    device = accelerator.device  # accelerator의 device 사용

    # 데이터 설정
    pid = find_option("pid", "0")
    train_eval = Train_Eval(config, accelerator)
    print(f'PID = {pid}')
    # 4. 학습 또는 평가
    model_dir = pid + '_' + str(config.years[0]) + '_' + str(config.years[-1])

    if "train" in config_name:
        if model_dir not in os.listdir("models"):
            os.mkdir("models/" + model_dir)
        for alg in algs:
            if alg.name not in os.listdir("models/" + model_dir):
                os.mkdir("models/" + model_dir + "/" + alg.name)

        # accelerator를 사용하여 학습 시작
        train_eval.train(algs, int(pid), model_dir)
    else:
        # 평가를 위해 데이터 준비
        train_eval.eval(algs, model_dir)

    # Alg_postal_temporal_Transformer 및 Postal_temporal_Transformer 클래스는 이전과 동일하게 정의
    # accelerate와 호환되도록 장치 배치 코드를 수정
