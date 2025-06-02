import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class LandscapeDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        """
        Custom dataset for landscape images
        root_dir: directory with all the images
        transform: pytorch transforms
        limit: limit number of images (for testing)
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files
        self.images = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        for file in os.listdir(root_dir):
            if file.lower().endswith(valid_extensions):
                self.images.append(file)
        
        # Limit dataset size if specified
        if limit:
            self.images = self.images[:limit]
        
        print(f"Found {len(self.images)} images in {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_transform(image_size=64):
    """Get preprocessing transform for images"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def preprocess_dataset(input_dir, output_dir, image_size=64, limit=None):
    """
    Preprocess high-res images to lower resolution
    Saves processed images to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if limit:
        images = images[:limit]
    
    print(f"Preprocessing {len(images)} images to {image_size}x{image_size}...")
    
    for img_name in tqdm(images, desc="Resizing images"):
        try:
            img_path = os.path.join(input_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_resized = transform(img)
            
            # Save with jpg to save space
            output_path = os.path.join(output_dir, img_name.rsplit('.', 1)[0] + '.jpg')
            img_resized.save(output_path, 'JPEG', quality=95)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    print(f"Preprocessing complete! Images saved to {output_dir}")


def save_samples(generator, fixed_noise, epoch, output_dir, device):
    """Generate and save sample images"""
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise.to(device))
        fake = (fake + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        
        # Save grid
        save_image(fake, os.path.join(output_dir, f'samples_epoch_{epoch}.png'), nrow=8)
    generator.train()


def plot_losses(g_losses, d_losses, save_path=None):
    """Plot generator and discriminator losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def create_gif(image_folder, output_path, duration=100):
    """Create GIF from generated samples"""
    import glob
    from PIL import Image
    
    # Get all sample images
    images = []
    files = sorted(glob.glob(os.path.join(image_folder, 'samples_epoch_*.png')))
    
    for filename in files:
        images.append(Image.open(filename))
    
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:], 
                      duration=duration, loop=0)
        print(f"GIF saved to {output_path}")
    else:
        print("No images found to create GIF") 