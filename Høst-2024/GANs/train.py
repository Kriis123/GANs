import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import datetime

from model import Generator, Discriminator, weights_init
from utils import (LandscapeDataset, get_transform, save_samples, 
                   plot_losses, create_gif, preprocess_dataset)


def train_gan(config):
    # Set device - M1 Mac uses 'mps'
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    
    # Create models
    netG = Generator(nz=config.nz, ngf=config.ngf, nc=3).to(device)
    netD = Discriminator(nc=3, ndf=config.ndf).to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    
    # Create dataset and dataloader
    transform = get_transform(config.image_size)
    dataset = LandscapeDataset(config.data_dir, transform=transform, limit=config.data_limit)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                          shuffle=True, num_workers=0)  # num_workers=0 for M1 compatibility
    
    # Fixed noise for consistent samples
    fixed_noise = torch.randn(64, config.nz, 1, 1, device=device)
    
    # Labels
    real_label = 1.0
    fake_label = 0.0
    
    # Training stats
    G_losses = []
    D_losses = []
    
    print("Starting Training...")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Image size: {config.image_size}")
    
    for epoch in range(config.num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for i, data in enumerate(progress_bar):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            
            # Train with real
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with fake
            noise = torch.randn(batch_size, config.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optimizerG.step()
            
            # Update stats
            epoch_g_loss += errG.item()
            epoch_d_loss += errD.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f'{errD.item():.4f}',
                'G_loss': f'{errG.item():.4f}',
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z1:.4f}/{D_G_z2:.4f}'
            })
            
            # Save training stats
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        
        # Save samples every few epochs
        if (epoch + 1) % config.sample_interval == 0:
            save_samples(netG, fixed_noise, epoch + 1, 
                        os.path.join(config.output_dir, 'samples'), device)
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
            }
            checkpoint_path = os.path.join(config.output_dir, 'checkpoints', 
                                         f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"\nCheckpoint saved at epoch {epoch + 1}")
    
    print("\nTraining complete!")
    
    # Save final models
    torch.save(netG.state_dict(), os.path.join(config.output_dir, 'generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(config.output_dir, 'discriminator_final.pth'))
    
    # Plot losses
    plot_losses(G_losses, D_losses, 
                save_path=os.path.join(config.output_dir, 'loss_plot.png'))
    
    # Create GIF
    create_gif(os.path.join(config.output_dir, 'samples'), 
               os.path.join(config.output_dir, 'training_progress.gif'))
    
    return netG, netD


def main():
    parser = argparse.ArgumentParser(description='Train DCGAN on Landscape Images')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess images to lower resolution')
    parser.add_argument('--preprocessed_dir', type=str, default='data/landscapes_64',
                        help='Directory to save preprocessed images')
    parser.add_argument('--data_limit', type=int, default=None,
                        help='Limit number of images to use (for testing)')
    
    # Model arguments
    parser.add_argument('--image_size', type=int, default=64,
                        help='Size of images (default: 64)')
    parser.add_argument('--nz', type=int, default=100,
                        help='Size of latent vector')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Number of generator filters')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Number of discriminator filters')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 for Adam optimizer')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Interval for saving samples')
    parser.add_argument('--checkpoint_interval', type=int, default=20,
                        help='Interval for saving checkpoints')
    
    args = parser.parse_args()
    
    # Preprocess dataset if requested
    if args.preprocess:
        print("Preprocessing dataset...")
        preprocess_dataset(args.data_dir, args.preprocessed_dir, 
                         args.image_size, args.data_limit)
        args.data_dir = args.preprocessed_dir
    
    # Train the GAN
    train_gan(args)


if __name__ == '__main__':
    main()
