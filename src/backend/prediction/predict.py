from PIL import Image
import torch
from torchvision import transforms

def predict(image_path, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to('cuda')

    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return class_names[predicted.item()]
