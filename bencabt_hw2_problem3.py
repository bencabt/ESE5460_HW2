# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from allcnn import allcnn_t
from torchvision.transforms import ToTensor
import torch.optim as optim
from datetime import datetime
import os
import numpy as np

#Key thinkgs to implement:
#Optimizer: 
# 1. LR .1 for first 40 epochs, .01 for next 40, .001 for last 20
# 2. SGD with momentum .9, weight decay 1e-3
#
# Transforms:
# 3. Data augmentation: mirror flips, cropping, and padding
# 
# Training
# 4. Dropout and batch-norm

MODEL_SAVE_PATH = "weights/allcnn_cifar10_20251022_195204.pth"

# Data loaders
# Customize data as it comes in 
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))])

#Download and prepare CIFAR-10 dataset
training_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

batch_size = 128
trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

#Check data shapes
for X,y in trainloader:
    #Batch, channels, height, width
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    X.permute(0,2,3,1)
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#Set up model
model = allcnn_t().to(device)
print(model)

#Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1) #assist w/ LR changes

#Training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() #set model to training mode

    #training loop
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backpropagation
        optimizer.zero_grad() #zero gradients
        loss.backward() #calculate gradients
        #grad clipped sourced from internet
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step() #update weights

        if batch % 100 == 0: #print out loss every 100 batches
            loss, current = loss.item(), batch * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#Validation
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #set model to evaluation mode
    test_loss, correct = 0, 0

    with torch.no_grad(): #no need to track gradients
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Check if the model file already exists
if os.path.exists(MODEL_SAVE_PATH):
    print(f"Weights found at {MODEL_SAVE_PATH}. Skipping training loop.")
    
    # Load the trained model state
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    
    # Set to evaluation mode for Problem 3(b) calculations
    model.eval()

else:
    # If the file is not found, commence training
    print("No existing weights found. COMMENCING TRAINING...")
    
    #Main training loop
    for epoch in range(100):
        print(f"Epoch {epoch+1}, lr={optimizer.param_groups[0]['lr']}")
        train(trainloader, model, loss_fn, optimizer)
        test(testloader, model, loss_fn)
        scheduler.step()
    print("Done!")

    #Saving the model
    #torch.save(model.state_dict(), "allcnn_cifar10.pth")
    #print("Saved PyTorch Model State to allcnn_cifar10.pth")
    output_dir = "/content/drive/MyDrive/ESE5460_HW2_outputs/"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"allcnn_cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), os.path.join(output_dir, filename))
    print("Saved model to Google Drive!")

###PART 3(b) 1###
#eval mode
model.eval()

# Get a batch of test data
X, y = next(iter(testloader))
X, y = X.to(device), y.to(device)
X.requires_grad = True #enable gradient tracking on input

output = model(X)  # Forward pass
pred = output.argmax(dim=1)  # Index of the max log-probability

# save correct and incorrect predictions
correct_predictions = (pred == y).nonzero(as_tuple=True)[0]
incorrect_predictions = (pred != y).nonzero(as_tuple=True)[0]

#dx calculation
loss = loss_fn(output, y)
loss.backward()
dx = X.grad.data.clone()


CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010])

#plotting dx for correct and incorrect predictions (first 3 of each), copilot supported plotting code
import matplotlib.pyplot as plt
def plot_image_and_gradient(X, dx, title, index):
    # Convert tensors to NumPy and handle channels/plotting
    X_np = X[index].detach().cpu().numpy().transpose(1, 2, 0) # C, H, W -> H, W, C
    dx_np = dx[index].cpu().numpy().transpose(1, 2, 0)

    X_denormalized = X_np * CIFAR_STD + CIFAR_MEAN
    X_denormalized = np.clip(X_denormalized, 0, 1)
    
    # Simple visualization: use the maximum absolute value across all channels
    dx_plot = np.max(np.abs(dx_np), axis=2) 
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title)

    # Plot Original Image
    axes[0].imshow(X_denormalized) # De-normalize CIFAR-10 image for viewing
    axes[0].set_title(f"Original:")
    axes[0].axis('off')
    
    # Plot Gradient (dx)
    # The gradient is the visualization of what direction in pixel space increases the loss.
    axes[1].imshow(dx_plot, cmap='viridis') 
    axes[1].set_title("Max Gradient (|dx|)")
    axes[1].axis('off')

    plt.show()

#PLOTTING
correct_idx = correct_predictions[0].item()
plot_image_and_gradient(X, dx, "Correctly Classified Sample", correct_idx)

incorrect_idx = incorrect_predictions[0].item()
plot_image_and_gradient(X, dx, "Incorrectly Classified Sample", incorrect_idx)

correct_idx = correct_predictions[1].item()
plot_image_and_gradient(X, dx, "Correctly Classified Sample", correct_idx)

incorrect_idx = incorrect_predictions[1].item()
plot_image_and_gradient(X, dx, "Incorrectly Classified Sample", incorrect_idx)


###PART 3(b) 2###
eps = 8.0 / 255.0  # Example step size
mini_batch_step_losses = torch.zeros(5).to(device)  
total_images_processed = 0

for x,y in zip(X, y):
    x = x.unsqueeze(0).clone().detach() #was not previously doing this, internet said to detach to avoid in-place errors
    y = y.unsqueeze(0)
    x.requires_grad = True
    for k in range(5):
        if x.grad is not None:
            x.grad.zero_()
        # forward propagate x through the network # backprop the loss
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        #  extract signed gradient
        dx = x.grad.data.clone()
        # perturb the image
        perturbation = eps * torch.sign(dx)
        x.data.add_(perturbation) # recommended via internet, previously had in place update
        
        with torch.no_grad():
            ell = loss_fn(model(x), y)
            
        mini_batch_step_losses[k] += ell.item()
    total_images_processed += 1

avg_step_losses = mini_batch_step_losses / total_images_processed
# Plot the loss on the perturbed images as a function of the number of steps 
# Move to CPU for NumPy/Matplotlib plotting
avg_step_losses_np = avg_step_losses.cpu().numpy()
steps = np.arange(1, 6) # Steps 1 through 5

plt.figure(figsize=(7, 5))
plt.plot(steps, avg_step_losses_np, marker='o', linestyle='-', color='red')

plt.title(f'{total_images_processed}-Image Avg Loss vs. Perturbation Steps')
plt.xlabel('Number of Perturbation Steps (k)')
plt.ylabel('Average Cross-Entropy Loss')
plt.xticks(steps) # Ensure ticks are at 1, 2, 3, 4, 5
plt.grid(True)
plt.show() 
