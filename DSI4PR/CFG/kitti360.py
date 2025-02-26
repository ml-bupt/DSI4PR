import torch
from pathlib import Path

class CFG:
    expid = Path(__file__).stem
    debug = False

    # data loader 
    dataloader = "KITTI360"
    fisheye = False
    data_path = "/home/zxs/ml/KITTI-360"
    train_sequences = ["0003", "0004", "0005", "0006", "0007", "0009", "0010"] #To set

    logdir = f"data/{expid}/log/"
    expdir = f"data/{expid}/"
    best_model_path = f"{expdir}best.pth"
    final_model_path = f"{expdir}model.pth"
    batch_size = 32 
    num_workers = 8
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
    # image size
    size = 224

    projection_dim = 256 
    dropout = 0.1
    wandbMode = "disabled" #online or disabled
    model = "DSI4PR"
    model_dim = 576
    num_heads = 8
    # Cropping
    crop_distance=True
    distance_threshold=50

    details = f"Exp Id: {expid} \nTraining Sequences: {train_sequences} \nBatch Size: {batch_size}"