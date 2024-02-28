from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch

data_dir = 'assignment-2-video-search\\3-embedding-model\\model-input-dataset'

# create dataloader with required transforms 
tc = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()              
    ])

image_datasets = datasets.ImageFolder(data_dir, transform=tc)
dloader = torch.utils.data.DataLoader(image_datasets, batch_size=10, shuffle=False)

print(len(image_datasets)) # returns 1150

# fetch pretrained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Select the desired layer
layer = model._modules.get('avgpool')

def copy_embeddings(m, i, o):
    """Copy embeddings from the penultimate layer.
    """
    o = o[:, :, 0, 0].detach().numpy().tolist()
    outputs.append(o)

outputs = []
# attach hook to the penulimate layer
_ = layer.register_forward_hook(copy_embeddings)

model.eval() # Inference mode

# Generate image's embeddings for all images in dloader and saves 
# them in the list outputs
for X, y in dloader:
    _ = model(X)
print(len(outputs)) #returns 115

# flatten list of embeddings to remove batches
list_embeddings = [item for sublist in outputs for item in sublist]

print(len(list_embeddings)) # returns 918
print(np.array(list_embeddings[0]).shape) #returns (512,)