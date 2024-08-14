import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

class DeepAppearanceDescriptor:
    def __init__(self):
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Identity()
        self.resnet.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image, bbox):
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        crop = self.transform(crop).unsqueeze(0)
        with torch.no_grad():
            features = self.resnet(crop)
        return features.squeeze().numpy()

    def cosine_distance(self, feat1, feat2):
        return 1 - np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))