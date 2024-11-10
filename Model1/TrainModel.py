import os
import numpy as np # type: ignore
import time
import torch
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from chess import pgn # type: ignore
from tqdm import tqdm # type: ignore


#Data preprocessing===================================================================================
def load_pgn(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

files = [file for file in os.listdir(r"/.idea/Chess/LichessEliteDatabase") if file.endswith(".pgn")]
LIMIT_OF_FILES = min(len(files), 15)
games = []
i = 1
for file in tqdm(files):
    games.extend(load_pgn(rf"C:\Users\Tuan Dung\IdeaProjects\CSCHTTT\.idea\Chess\LichessEliteDatabase\{file}"))
    if i > LIMIT_OF_FILES:
        break
    i += 1

print(f"GAMES PARSED: {len(games)}")


#Convert data into tensors================================================================================
from auxiliary_func import create_input_for_nn, encode_moves
X, y = create_input_for_nn(games)

print(f"NUMBER OF SAMPLES: {len(y)}")
X = X[0:2500000]
y = y[0:2500000]
y, move_to_int = encode_moves(y)
num_classes = len(move_to_int)
print(f"NUMBER OF CLASSES: {num_classes}")
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)



from dataset import ChessDataset
from model import ChessModel
# Create Dataset and DataLoader
dataset = ChessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model Initialization
model = ChessModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Training=====================================================================================
num_epochs = 100
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        outputs = model(inputs)  # Raw logits

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()
    end_time = time.time()
    epoch_time = end_time - start_time
    minutes: int = int(epoch_time // 60)
    seconds: int = int(epoch_time) - minutes * 60
    print(f'Epoch {epoch + 1 + 100}/{num_epochs + 1 + 100}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s')


#Save Model===================================================================================
torch.save(model.state_dict(), "models/TORCH_100EPOCHS.pth")
import pickle

with open("models/heavy_move_to_int", "wb") as file:
    pickle.dump(move_to_int, file)