import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from model import Generator

def generate_single_image(generator, device, nz=100, seed=None):
    """Generate a single landscape image"""
    generator.eval()
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(1, nz, 1, 1, device=device)
        
        # Generate image
        fake_image = generator(noise)
        
        # Denormalize from [-1, 1] to [0, 1]
        fake_image = (fake_image + 1) / 2
        
        # Convert to PIL image
        fake_image = fake_image.squeeze(0).cpu()
        transform = transforms.ToPILImage()
        pil_image = transform(fake_image)
        
    return pil_image

def main():
    parser = argparse.ArgumentParser(description='Generate single landscape images')
    parser.add_argument('--model_path', type=str, default='outputs/generator_final.pth',
                        help='Path to trained generator model')
    parser.add_argument('--output_dir', type=str, default='outputs/single_images',
                        help='Directory to save generated images')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to generate')
    parser.add_argument('--nz', type=int, default=100,
                        help='Size of latent vector')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Number of generator filters')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load generator
    generator = Generator(nz=args.nz, ngf=args.ngf, nc=3).to(device)
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()
    
    print(f"Generating {args.num_images} images...")
    
    # Generate images
    for i in range(args.num_images):
        # Use different seed for each image
        seed = args.seed + i if args.seed is not None else None
        
        # Generate image
        image = generate_single_image(generator, device, args.nz, seed)
        
        # Save image
        filename = f'landscape_{i+1:04d}.png'
        filepath = os.path.join(args.output_dir, filename)
        image.save(filepath)
        print(f"Saved: {filepath}")
    
    print(f"\nGenerated {args.num_images} images in {args.output_dir}")

if __name__ == '__main__':
    main()