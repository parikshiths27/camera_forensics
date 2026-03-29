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
        
    predictions = preds.cpu().numpy()
    probs = probabilities.cpu().numpy()
    patch_confidences = [probs[i][predictions[i]] for i in range(len(predictions))]
    
    # Majority voting
    vote_counts = Counter(predictions)
    majority_class_idx, vote_freq = vote_counts.most_common(1)[0]
    
    # Calculate average confidence strictly for the patches that voted for the majority class
    majority_confidences = [patch_confidences[i] for i, pred in enumerate(predictions) if pred == majority_class_idx]
    avg_confidence = np.mean(majority_confidences) if majority_confidences else 0.0
    
    # === ANOMALY DETECTION (Open-Set Recognition) ===
    # HACKATHON DEMO PARANOIA MODE:
    # Because iOS browsers covertly apply harsh JPEG compression to live photo uploads,
    # we are artificially cranking our security thresholds to the absolute maximum.
    # If the hardware static does not reach a staggering 90% confidence score,
    # OR if the "AI Jury" doesn't reach a near-unanimous 12 out of 16 agreement,
    # the system will instantly reject the photo as a corrupted anomaly!
    if vote_freq < 12 or avg_confidence < 0.90:
        # We flag it as an Unknown Device / Corrupted Forensic Evidence
        return "Unknown Device (Not in Database)", float(avg_confidence)

    pred_class = class_mapping[majority_class_idx]
    
    return pred_class, float(avg_confidence)
