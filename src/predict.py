import torch
import numpy as np
from collections import Counter
from src.model import CameraForensicsCNN
from src.patch_generation import process_pipeline

def predict_camera(image_path, model_path, class_mapping_path):
    """
    Runs the inference pipeline on an image path to predict the camera model.
    """
    # Load class mapping
    class_mapping = {}
    with open(class_mapping_path, 'r') as f:
        for line in f:
            idx, name = line.strip().split(',')
            class_mapping[int(idx)] = name
            
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CameraForensicsCNN(num_classes=len(class_mapping)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Process pipeline (Preprocess -> Residual Extraction -> Patch Generation)
    patches = process_pipeline(image_path) # numpy (16, 32, 32)
    patches_tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(patches_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
    preds = preds.cpu().numpy()
    probs = probabilities.cpu().numpy()
    
    # Majority voting over the 16 patches
    counter = Counter(preds)
    most_common_class_idx, count = counter.most_common(1)[0]
    
    # Extracted final class prediction
    predicted_class_name = class_mapping[most_common_class_idx]
    
    # Calculate confidence: Average probability of the predicted class across patches that derived it
    voting_indices = [i for i, p in enumerate(preds) if p == most_common_class_idx]
    avg_confidence = np.mean([probs[i][most_common_class_idx] for i in voting_indices])
    
    patch_predictions = {class_mapping[idx]: prob for idx, prob in zip(preds, probs)}
    
    return predicted_class_name, float(avg_confidence)
