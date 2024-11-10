import torch
import torch.optim as optim
from torch.utils.data import DataLoader


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
LIMIT_OF_FILES = min(len(files), 3)
new_games = []
i = 1
for file in tqdm(files):
    new_games.extend(load_pgn(rf"C:\Users\Tuan Dung\IdeaProjects\CSCHTTT\.idea\Chess\LichessEliteDatabase\{file}"))
    if i > LIMIT_OF_FILES:
        break
    i += 1

print(f"GAMES PARSED: {len(games)}")

# Tải mô hình đã huấn luyện trước
model = ChessModel(num_classes=num_classes)
model.load_state_dict(torch.load("models/TORCH_100EPOCHS.pth"))
model.to(device)


# Chuyển đổi dữ liệu mới thành tensor
X_new, y_new = create_input_for_nn(new_games)
y_new, move_to_int_new = encode_moves(y_new)  # Đảm bảo rằng encoding phải đồng bộ với mô hình gốc

X_new = torch.tensor(X_new, dtype=torch.float32)
y_new = torch.tensor(y_new, dtype=torch.long)

# Tạo DataLoader mới
new_dataset = ChessDataset(X_new, y_new)
new_dataloader = DataLoader(new_dataset, batch_size=64, shuffle=True)

# Khởi tạo bộ tối ưu và tiếp tục huấn luyện
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Giảm learning rate

num_fine_tune_epochs = 20  # Số epoch để huấn luyện tiếp
for epoch in range(num_fine_tune_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(new_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_fine_tune_epochs}, Loss: {running_loss / len(new_dataloader):.4f}')

# Lưu lại mô hình sau khi fine-tune
torch.save(model.state_dict(), "models/TORCH_FINE_TUNED.pth")
