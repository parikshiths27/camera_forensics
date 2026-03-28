import os
import sys
# Make sure the project root is in sys.path so 'src' module is recognizable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path
from src.model import CameraForensicsCNN
from src.patch_generation import process_pipeline
import copy

class CameraForensicsDataset(Dataset):
    def __init__(self, dataset_dir, is_train=False):
        self.dataset_dir = Path(dataset_dir)
        self.is_train = is_train
        self.classes = sorted([d.name for d in self.dataset_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Add basic augmentations for training data
        self.augmentation = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # Patches are already tensors
        ]) if is_train else None
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.dataset_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                    
        print(f"Extracting patches from {len(self.samples)} images...")
        self.patches = []
        self.labels = []
        for img_path, label in self.samples:
            try:
                img_patches = process_pipeline(img_path)
                for p in img_patches:
                    self.patches.append(p)
                    self.labels.append(label)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
                
        self.patches = torch.tensor(self.patches, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print(f"Total patches loaded: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.augmentation:
            patch = self.augmentation(patch)
        return patch, self.labels[idx]

def train_model(data_dir=r"c:\Projects\camera_forensics\dataset", epochs=50, batch_size=64, lr=0.001):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    print("Loading datasets...")
    train_dataset = CameraForensicsDataset(train_dir, is_train=True)
    test_dataset = CameraForensicsDataset(test_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CameraForensicsCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Deep copy the model if it has the best accuracy so far
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  --> Saved new best model with Val Acc: {best_acc:.4f}")
            
    # Save the absolute best model
    os.makedirs(r"c:\Projects\camera_forensics\models", exist_ok=True)
    model.load_state_dict(best_model_wts)
    model_path = r"c:\Projects\camera_forensics\models\forensics_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")
    print(f"Best model saved to {model_path}")

    with open(r"c:\Projects\camera_forensics\models\class_mapping.txt", "w") as f:
        for idx, cls in enumerate(train_dataset.classes):
            f.write(f"{idx},{cls}\n")

if __name__ == "__main__":
    train_model(epochs=100, batch_size=32)
