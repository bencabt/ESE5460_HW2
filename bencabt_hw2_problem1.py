import torch
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import v2
from torchvision import transforms 
from torchvision.transforms import functional as F

##1C##
# Load ResNet-18
resnet18 = models.resnet18(pretrained=False)

# Iterate over all named parameters
for name, param in resnet18.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | #Params: {param.numel()}")

#Total parameters
total_params = sum(p.numel() for p in resnet18.parameters())
print(f"Total parameters: {total_params}")

##1E SEPERATED PARAMS##
bn_params = []
bias_params = []
remaining_params = []
for name, param in resnet18.named_parameters():
    if 'bn' in name:
        bn_params.append((name, param))
    elif 'bias' in name:
        bias_params.append((name, param))
    else:
        remaining_params.append((name, param))


##Copilot auto generated the following print statements for separated parameters##
print("\nBatchNorm Parameters:")
for name, param in bn_params:
    print(f"Layer: {name} | Size: {param.size()} | #Params: {param.numel()}")

print("\nBias Parameters:")
for name, param in bias_params:
    print(f"Layer: {name} | Size: {param.size()} | #Params: {param.numel()}")

print("\nRemaining Parameters:")
for name, param in remaining_params:
    print(f"Layer: {name} | Size: {param.size()} | #Params: {param.numel()}")


##1F Transforms##
#From pytorch example
plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(0)

IMAGE_PATH = "data/astronaut.jpg"
orig_image = Image.open(IMAGE_PATH)

image_list = []

# ShearX
shear_x = v2.RandomAffine(degrees=0, shear=(0, 40, 0, 0))
image_list.append(shear_x(orig_image))

# ShearY
shear_y = v2.RandomAffine(degrees=0, shear=(0, 0, 0, 40))
image_list.append(shear_y(orig_image))

# TranslateX
translate_x = v2.RandomAffine(degrees=0, translate=(0.5, 0))
image_list.append(translate_x(orig_image))

# TranslateY
translate_y = v2.RandomAffine(degrees=0, translate=(0, 0.5))
image_list.append(translate_y(orig_image))

# Rotate
rotate = transforms.RandomRotation(degrees=45)
image_list.append(rotate(orig_image))

# Brightness
brightness = v2.ColorJitter(brightness=.9)
image_list.append(brightness(orig_image))

# Color
color = v2.ColorJitter(saturation=.9)
image_list.append(color(orig_image))

# Contrast
contrast = v2.ColorJitter(contrast=.9)
image_list.append(contrast(orig_image))

# Sharpness
sharpness = v2.RandomAdjustSharpness(sharpness_factor=2, p=1.0)
image_list.append(sharpness(orig_image))

# Posterize
posterize = v2.RandomPosterize(bits=8, p=1.0)
image_list.append(posterize(orig_image))

# Solarize 
solarize = v2.RandomSolarize(threshold=128)
image_list.append(solarize(orig_image))

# Equalize
equalize = v2.RandomEqualize(1.0)
image_list.append(equalize(orig_image))


#plotting all transformed images
titles = ['Original', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 
          'Brightness', 'Color', 'Contrast', 'Sharpness', 'Posterize', 'Solarize', 'Equalize']

#copilot generated
plt.figure(figsize=(15, 10))
for i, img in enumerate([orig_image] + image_list):
    plt.subplot(3, 5, i + 1)
    plt.imshow(img)
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

 
