import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataset import ChessDataset
from model import ChessModel
from ProcessDataWithChunks import load_pgn, build_move_dict
import pickle

def train_model_in_chunks(model, optimizer, criterion, save_dir="D:/DevGame/ChessDataSet", num_epochs=100, batch_size=64, device="cpu", checkpoint_interval=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        chunk_index = 0

        # Train through each chunk of data
        while True:
            # Load a chunk of data
            X_path = f"{save_dir}/X_chunk_{chunk_index}.npy"
            y_path = f"{save_dir}/y_chunk_{chunk_index}.npy"
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                break  # Exit if no more chunks

            # Load the chunk data into memory
            X_chunk = np.load(X_path)
            y_chunk = np.load(y_path)

            # Convert numpy arrays to PyTorch tensors
            X_chunk = torch.tensor(X_chunk, dtype=torch.float32).to(device)
            y_chunk = torch.tensor(y_chunk, dtype=torch.long).to(device)

            # Create DataLoader for the current chunk
            dataset = ChessDataset(X_chunk, y_chunk)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Train on the current chunk
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1} Chunk {chunk_index+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update parameters
                optimizer.step()
                running_loss += loss.item()

            chunk_index += 1  # Move to the next chunk

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / (chunk_index * 157):.4f}")




# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load num_classes
move_dict = np.load("D:/DevGame/ChessDataSet/move_dict.npy", allow_pickle=True).item()
num_classes = len(move_dict)
print(f"Calculated num_classes: {num_classes}")

# Model Initialization
model = ChessModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Load checkpoint if resuming from a specific epoch
checkpoint_path = "models/TORCH_checkpoint_epoch_20_index_1.pth"  # Change to your desired checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0

epochs = 10
# Train the model in chunks with checkpoint saving
#train_model_in_chunks(model, optimizer, criterion, save_dir="D:/DevGame/ChessDataSet", num_epochs=epochs, batch_size=64, device=device)


checkpoint_path = f"models/TORCH_checkpoint_epoch_{start_epoch + epochs}.pth"
torch.save({
    'epoch': start_epoch + epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, checkpoint_path)
print(f"Checkpoint saved at epoch {start_epoch+start_epoch}: {checkpoint_path}")

# Final save of model and move_dict
if(start_epoch + start_epoch == 100):
    torch.save(model.state_dict(), "models/TORCH_final_model_100EPOCHS.pth")
with open("models/heavy_move_to_int_1", "wb") as file:
    pickle.dump(move_dict, file)
