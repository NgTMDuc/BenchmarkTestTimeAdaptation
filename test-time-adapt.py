import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(f"Using {device}: {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM" if device == "cuda" else "Using CPU")

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# Large model (ResNet-152)
model = models.resnet152(weights=None).to(device)
model = torch.compile(model)  # Optimize execution (PyTorch 2.0+)

# Generate large synthetic dataset (512x3x512x512)
batch_size = 64  # Large batch to push memory usage
image_size = (3, 512, 512)
num_samples = 20000  # Large dataset

x = torch.randn(num_samples, *image_size)
y = torch.randint(0, 1000, (num_samples,))  # 1000 classes like ImageNet
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()

# Training loop (5 epochs)
for epoch in range(5):
    model.train()
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        with autocast():  # Mixed precision
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Check GPU memory usage
print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
print(f"Memory Cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
