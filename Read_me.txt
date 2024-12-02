1. 설치
    python 3.10.12
    pip install -r requirements.txt

2. 실행
    python train_eval.py config={config_name} pid=0 device=0
        - config: 실험 setup
            years: 무슨년도 데이터 사용할지 (오름차순으로 list 형태)
            batch_size: batch_size
            training_mode: Training시 true / Eval시 False
            start_epoch: 만약 이어서 학습한다면 불러올 epoch
            num_epochs: 최대 Training epoch
            lr: train시 learning rate
            epoch: eval시 몇번째 epoch의 모델을 사용할지
            algorithm = 알고리즘파이썬파일명.알고리즘class명('알고리즘이름(지정가능)')

        - pid: seed설정시 사용 (연도가 같아도 시드가 다를시 다른 실험) DEFAULT=0
        - device: cuda사용시 몇번째 cuda DEFAULT=0

