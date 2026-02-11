import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["Healthy", "Moderate / Stressed", "Unhealthy / Diseased"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = models.mobilenet_v2(pretrained=False)

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 3)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


def predict_image(model, image: Image.Image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)[0]

    confidences = probs.cpu().numpy()
    predicted_idx = confidences.argmax()

    return {
        "label": CLASS_NAMES[predicted_idx],
        "confidence": float(confidences[predicted_idx]),
        "all_confidences": confidences
    }