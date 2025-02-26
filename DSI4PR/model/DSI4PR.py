import timm
import torch.nn.functional as F
from torch import nn
import torch

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    This is for crossvit_small_240
    """

    def __init__(
        self, model_name, pretrained, trainable
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0)


        for p in self.model.parameters():
            p.requires_grad = trainable
  
    def forward(self, x):
        y1 = self.model(x)
        return y1

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(CrossAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, query, key, value):
        query = query.unsqueeze(0)  #  [1, batch_size, embed_dim]
        key = key.unsqueeze(0)      #  [1, batch_size, embed_dim]
        value = value.unsqueeze(0)  #  [1, batch_size, embed_dim]

        # Compute cross-attention
        attn_output, _ = self.self_attn(query, key, value)

        # Add & Norm
        output = self.norm(query + attn_output)

        # Feed-forward layer
        output = self.fc(self.activation(output))
        output = self.dropout(output)

        output = output.squeeze(0)  
   
        return output



class ProjectionT2(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
    ):
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.device=CFG.device
        self.temperature=CFG.temperature
        self.encoder_camera = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable)
        self.encoder_lidar = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable)
        self.crossattention = CrossAttention(embed_dim=CFG.model_dim, num_heads=CFG.num_heads, dropout=CFG.dropout)
        self.projection_lidar = ProjectionT2(embedding_dim=CFG.image_embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout)
        self.projection_camera = ProjectionT2(embedding_dim=CFG.image_embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout)
        #ProjectionT1
        self.fc = nn.Linear(576, 2048)  
        self.relu = nn.ReLU() 
    def forward(self, batch):
        # Getting camera Image and lidar range image Features
        camera_image_features = self.encoder_camera(batch["depth_image"])
        lidar_image_features = self.encoder_lidar(batch["lidar_image"])

        camera_image_features = self.crossattention(query=camera_image_features, key=lidar_image_features, value=lidar_image_features)
        lidar_image_features =self.crossattention(query=lidar_image_features, key=camera_image_features, value=camera_image_features)
        #ProjectionTransformer
        #T1
        camera_image_features = self.fc(camera_image_features)
        lidar_image_features = self.fc(lidar_image_features)
        camera_image_features = self.relu(camera_image_features)
        lidar_image_features = self.relu(lidar_image_features)
        #T2
        camera_image_embeddings = self.projection_camera(camera_image_features)
        lidar_image_embeddings = self.projection_lidar(lidar_image_features)

        # Calculating the Loss
        logits = (lidar_image_embeddings @ camera_image_embeddings.T) / self.temperature
        camera_similarity = camera_image_embeddings @ camera_image_embeddings.T
        lidar_similarity = lidar_image_embeddings @ lidar_image_embeddings.T
        targets = F.softmax(
            (camera_similarity + lidar_similarity) / 2 * self.temperature, dim=-1
        )
        lidar_loss = cross_entropy(logits, targets, reduction='none')
        camera_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (camera_loss + lidar_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def get_camera_embeddings(self, batch):

        camera_image_features = self.encoder_camera(batch["depth_image"].to(self.device))
        lidar_image_features = self.encoder_lidar(batch["lidar_image"].to(self.device))
        

        camera_image_features = self.crossattention(query=camera_image_features, key=lidar_image_features, value=lidar_image_features)
        lidar_image_features =self.crossattention(query=lidar_image_features, key=camera_image_features, value=camera_image_features)
       
        camera_image_features = self.fc(camera_image_features)
        camera_image_features = self.relu(camera_image_features)
        camera_image_embeddings = self.projection_camera(camera_image_features)
        
        return camera_image_embeddings

    def get_lidar_embeddings(self, batch):

        lidar_image_features = self.encoder_lidar(batch["lidar_image"].to(self.device))
        camera_image_features = self.encoder_camera(batch["depth_image"].to(self.device))
        
        camera_image_features = self.crossattention(query=camera_image_features, key=lidar_image_features, value=lidar_image_features)
        lidar_image_features =self.crossattention(query=lidar_image_features, key=camera_image_features, value=camera_image_features)

        lidar_image_features = self.fc(lidar_image_features)
        lidar_image_features = self.relu(lidar_image_features)
        lidar_image_embeddings = self.projection_lidar(lidar_image_features)
        
        return lidar_image_embeddings


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def get_topk(query_image_embeddings, lidar_image_embeddings, n=1):
    dot_similarity = query_image_embeddings @ lidar_image_embeddings.T
    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    return values, indices
