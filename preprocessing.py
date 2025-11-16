import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs("images", exist_ok=True)

# Load CIFAR-10 sample
dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True
)

original_img, _ = dataset[0]

# Define transforms
transform_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_flip = transforms.RandomHorizontalFlip(p=1.0)
transform_rotation = transforms.RandomRotation(20)
transform_crop = transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0))
transform_blur = transforms.GaussianBlur(kernel_size=5)

# Apply transforms
normalized_img = transform_normalize(original_img)
flipped_img = transform_flip(original_img)
rotated_img = transform_rotation(original_img)
cropped_img = transform_crop(original_img)
blurred_img = transform_blur(original_img)

# Convert normalized tensor back to image for viewing
def denormalize(img_tensor):
    img_tensor = img_tensor * 0.5 + 0.5
    return transforms.ToPILImage()(img_tensor)

normalized_img_pil = denormalize(normalized_img)

# Save results
original_img.save("images/original.png")
flipped_img.save("images/flip.png")
rotated_img.save("images/rotated.png")
cropped_img.save("images/cropped.png")
blurred_img.save("images/blurred.png")
normalized_img_pil.save("images/normalized.png")

print("Images saved inside /images folder.")
