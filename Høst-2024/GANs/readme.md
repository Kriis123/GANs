# 🏔️ Landscape GAN - Quick Sprint Project

Generate synthetic landscape images using GANs in 2-3 days on MacBook Pro M1.

## 🎯 Goal

Get a working GAN that generates recognizable landscape images. Perfect is the enemy of done - we want results fast!

## 📊 Dataset

**LHQ-1024 Dataset**
- 90,000 landscape images (1024×1024)
- [Download from Kaggle](https://www.kaggle.com/datasets/dimensi0n/lhq-1024)
- We'll resize to 64×64 for fast training

## 🏃‍♂️ 2-3 Day Sprint Plan

### Day 1: Setup & Basic Training
**Morning (3-4 hours)**
- [ ] Download dataset (subset of 10,000 images to start)
- [ ] Set up environment (PyTorch with M1 support)
- [ ] Preprocess images to 64×64
- [ ] Implement basic DCGAN from scratch

**Afternoon (3-4 hours)**
- [ ] Create data loader
- [ ] Start training
- [ ] Debug issues
- [ ] See first (probably terrible) results

### Day 2: Get It Working
**Morning (3-4 hours)**
- [ ] Fix training issues from Day 1
- [ ] Tune hyperparameters (learning rate, batch size)
- [ ] Implement proper loss logging
- [ ] Add checkpointing to save progress

**Afternoon (3-4 hours)**
- [ ] Continue training with better settings
- [ ] Generate sample grids every N iterations
- [ ] Try different random seeds
- [ ] Get recognizable landscape shapes

### Day 3: Polish & Results
**Morning (3-4 hours)**
- [ ] Final training run with best settings
- [ ] Create visualization of results
- [ ] Generate a grid of best samples
- [ ] Make a GIF of training progression

**Afternoon (2-3 hours)**
- [ ] Clean up code
- [ ] Document what worked/didn't work
- [ ] Create simple demo script
- [ ] Plan improvements for future

## 💻 Technical Details

### Simple DCGAN Architecture
```python
# Generator: 100 -> 64x64x3
# Input: 100-dim noise vector
# Output: 64x64 RGB image

# Discriminator: 64x64x3 -> 1
# Input: 64x64 RGB image  
# Output: Real/Fake classification
```

### M1 Pro Settings
```python
# Conservative settings that should work
batch_size = 64
learning_rate = 0.0002
image_size = 64
num_epochs = 100  # ~2-3 hours on M1
```

## 📁 Minimal Project Structure
```
landscape-gan-sprint/
├── data/
│   └── landscapes_64/     # Preprocessed 64x64 images
├── train.py              # All-in-one training script
├── model.py              # DCGAN architecture
├── utils.py              # Helper functions
├── outputs/
│   ├── samples/          # Generated images
│   └── checkpoints/      # Saved models
└── requirements.txt
```

## 🛠️ Quick Setup

```bash
# Install dependencies
pip install torch torchvision pillow matplotlib tqdm

# Download subset of data
# Resize to 64x64
# Start training
python train.py
```

## ✅ Success Criteria

By end of Day 3, you should have:
1. A trained model that generates 64×64 images
2. Images that look roughly like landscapes (sky, ground, maybe mountains)
3. Understanding of how GANs work in practice
4. Ideas for improvements

## 🚀 Future Improvements (If Time Allows)

Once the basic model works:
- Increase resolution to 128×128
- Try Progressive GAN for better quality
- Add conditional generation (day/night, seasons)
- Implement better evaluation metrics
- Train on full 90k dataset

## 📝 Key Lessons Expected

- GANs are unstable - expect training crashes
- Mode collapse is real - all landscapes might look similar
- Learning rates matter A LOT
- Batch size affects quality
- First results will look bad, that's normal

## 🎮 Minimum Viable Demo

```python
# Generate 16 random landscapes
noise = torch.randn(16, 100, 1, 1)
fake_landscapes = generator(noise)
save_image_grid(fake_landscapes, "my_landscapes.png")
```

---

**Remember**: This is a learning project. Getting any recognizable landscape is a win! Focus on understanding the process rather than perfect results.