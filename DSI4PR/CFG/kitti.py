import torch
from pathlib import Path

class CFG:
    expid = Path(__file__).stem
    #zero-shot transfer analysis
    # dataloader = "KITTI360"
    # fisheye = False
    # data_path = "/home/zxs/ml/KITTI-360"
    dataloader = "KITTI"
    data_path = "/home/zxs/ml/KITTI/data/dataset/sequences" #To set
    debug = False
    train_sequences =  [ "00"]
    expdir = f"data/{expid}/"
    best_model_path = f"{expdir}best.pth"
    final_model_path = f"{expdir}model.pth"
    batch_size = 32
    num_workers = 4 
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 70
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_image_model_name = 'crossvit_small_240'
    # image size
    size = 224
    image_embedding_dim = 2048

    max_length = 200
    pretrained = True
    trainable = True 
    temperature = 1.0
    # for projection head; used for both image and text encoders
    projection_dim = 256 
    model_dim = 576
    num_heads = 8
    dropout = 0.1
    model = "DSI4PR"
    crop_distance=True
    distance_threshold=50
    logdir = f"data/{expid}/log/"
    details = f"Exp Id: {expid} \nTraining on: {train_sequences} \nBatch Size: {batch_size}"
