import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance
import os
import argparse
from model import Generator

def enhance_image(image):
    """Apply post-processing to sharpen the image"""
    # Convert to PIL if tensor
    if isinstance(image, torch.Tensor):
        transform = transforms.ToPILImage()
        image = transform(image)
    
    # 1. Apply unsharp mask for sharpening
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # 2. Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    
    # 3. Enhance color
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.2)
    
    # 4. Slight sharpness enhancement
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    return image

def generate_enhanced_image(generator, device, nz=100, size=128):
    """Generate and enhance a single image with upscaling"""
    generator.eval()
    
    with torch.no_grad():
        # Generate image
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_image = generator(noise)
        
        # Denormalize
        fake_image = (fake_image + 1) / 2
        fake_image = fake_image.squeeze(0).cpu()
        
        # Convert to PIL
        transform = transforms.ToPILImage()
        image = transform(fake_image)
        
        # Upscale using LANCZOS (better than default)
        if size > 64:
            image = image.resize((size, size), Image.LANCZOS)
        
        # Apply enhancement
        image = enhance_image(image)
        
    return image

def main():
    parser = argparse.ArgumentParser(description='Generate enhanced landscape images')
    parser.add_argument('--model_path', type=str, default='outputs/generator_final.pth',
                        help='Path to trained generator model')
    parser.add_argument('--output_dir', type=str, default='outputs/enhanced_images',
                        help='Directory to save enhanced images')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to generate')
    parser.add_argument('--size', type=int, default=256,
                        help='Output image size (will upscale)')
    parser.add_argument('--grid', action='store_true',
                        help='Also create a grid of images')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load generator
    generator = Generator(nz=100, ngf=64, nc=3).to(device)
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()
    
    print(f"Generating {args.num_images} enhanced images at {args.size}x{args.size}...")
    
    images = []
    for i in range(args.num_images):
        # Generate enhanced image
        image = generate_enhanced_image(generator, device, size=args.size)
        
        # Save individual image
        filename = f'landscape_enhanced_{i+1:04d}.png'
        filepath = os.path.join(args.output_dir, filename)
        image.save(filepath, 'PNG', quality=100)
        print(f"Saved: {filepath}")
        
        images.append(image)
    
    # Create grid if requested
    if args.grid and len(images) >= 4:
        grid_size = min(4, int(len(images) ** 0.5))
        grid_image = Image.new('RGB', (args.size * grid_size, args.size * grid_size))
        
        for idx, img in enumerate(images[:grid_size*grid_size]):
            x = (idx % grid_size) * args.size
            y = (idx // grid_size) * args.size
            grid_image.paste(img, (x, y))
        
        grid_path = os.path.join(args.output_dir, 'enhanced_grid.png')
        grid_image.save(grid_path)
        print(f"\nGrid saved: {grid_path}")
    
    print(f"\nEnhanced all images! Check {args.output_dir}")

if __name__ == '__main__':
    main()