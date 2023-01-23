
import os 
from model import ResNet
import torch 

def get_models():
    path = "/Users/jcheigh/ML-Projects/Solving the Rubik's Cube with Deep RL/Models"
    target_path = "%s/%s/" % (path, "target")
    current_path = "%s/%s/" % (path, "current")
    model = None
    #make path for both 
    if os.path.exists(path):
        current_model = torch.load(current_path)
        target_model = torch.load(target_path)
    else:
        current_model = ResNet()
        target_model = ResNet()
    current_model.eval()
    target_model.eval()
    return current_model, target_model
