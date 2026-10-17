import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
from config import PROJECT_DIR

print("Loading vision models...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# one YOLO instance per camera (tracker state is kept inside the model)
yolo_cam1 = YOLO(str(PROJECT_DIR / "yolov8n.pt"))
yolo_cam2 = YOLO(str(PROJECT_DIR / "yolov8n.pt"))

resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.eval()
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(DEVICE)

preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Distinct colors for different matched pairs (BGR format)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
]
