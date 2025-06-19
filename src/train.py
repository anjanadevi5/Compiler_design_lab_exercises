import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import os
import copy

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, model_save_path='best_model.pth'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved to {model_save_path} with accuracy: {best_acc:.4f}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # --- Configuration ---
    data_dir =  '../dataset' # Path to the directory containing 'train' and 'val' folders
    num_classes = 2  # For 'passport' and 'dl'
    
    # MODIFIED for very small dataset:
    batch_size = 2  # Or 1, or up to the number of images in your smallest class
    num_epochs = 10 # Drastically reduce epochs to prevent quick overfitting
    
    learning_rate = 0.001
    model_save_path = 'resnet50_passport_dl_classifier.pth'

    # Set device to CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Data Augmentation and Normalization ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            # Optional: Add more augmentations
            # transforms.RandomRotation(degrees=5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- Load Data ---
    # Assumes data_dir has 'train' and 'val' subdirectories.
    # 'train' and 'val' subdirectories should each have subdirectories for each class (e.g., 'passport', 'dl')
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        
        # Ensure num_workers is appropriate. If issues arise, try num_workers=0.
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                     shuffle=True if x == 'train' else False, num_workers=2,
                                                     pin_memory=False) # pin_memory=False for CPU
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        print(f"Classes found: {class_names}")

        if len(class_names) != num_classes:
            print(f"Warning: Expected {num_classes} classes but found {len(class_names)} in the dataset.")
            num_classes = len(class_names) # Adjust if necessary

    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure your dataset is structured correctly in '{data_dir}'.")
        print("Expected structure: ")
        print(f"{data_dir}/train/passport/...")
        print(f"{data_dir}/train/dl/...")
        print(f"{data_dir}/val/passport/...")
        print(f"{data_dir}/val/dl/...")
        exit()


    # --- Model Setup ---
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all the network except the final layer (optional, for transfer learning)
    # for param in model_ft.parameters():
    #     param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    # --- Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    # To optimize only the classifier layer (if you froze other layers):
    # optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=learning_rate, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # --- Train and Evaluate ---
    print("Starting training...")
    trained_model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                dataloaders, dataset_sizes, device, num_epochs=num_epochs,
                                model_save_path=model_save_path)

    print("Training finished.")
    print(f"Best model saved to: {model_save_path}")
