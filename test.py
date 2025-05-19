import torch
from config import cfg
from models.bird_model import BirdCLEFModel
checkpoint = torch.load('output/model_fold0_best.pth', map_location=torch.device(cfg.device), weights_only=False)
model = BirdCLEFModel(cfg)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(cfg.device)
print(checkpoint.keys())