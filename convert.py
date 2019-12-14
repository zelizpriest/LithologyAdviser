import torch
from torchvision.models import mobilenet_v2

model = mobilenet_v2(pretrained=False)

weights = torch.load('models/Mobile_net_core.pkl')

model.load_state_dict(weights)

script_model = torch.jit.trace(model,input_tensor)
script_model.save("mobilenet-v2.pt")