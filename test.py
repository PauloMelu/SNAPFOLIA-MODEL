import torch
import torch.nn as nn
# from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import datasets
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_CLASSES = 3

# rebuild model
# weights = EfficientNet_B3_Weights.DEFAULT
# model = efficientnet_b3(weights=None)  # ⚠️ no pretrained weights here

weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)


model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES  # 👈 must match training
)

model.load_state_dict(torch.load("efficientnet_b3_test.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

train_ds = datasets.ImageFolder("dataset/train")
class_names = train_ds.classes

transform = transforms.Compose([
    transforms.Resize((300, 300)),  # or 224 if you changed it
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img = Image.open("test.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to(DEVICE)  # add batch dimension


with torch.no_grad():
    out = model(x)
    pred = out.argmax(1).item()


probs = F.softmax(out, dim=1)
confidence = probs[0][pred].item()

print(probs)
print(f"Prediction: {class_names[pred]} ({confidence:.2%})")