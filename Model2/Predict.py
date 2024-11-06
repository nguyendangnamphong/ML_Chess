from chess import Board, pgn
from auxiliary_func import board_to_matrix
import torch
from model import ChessModel
import pickle
import numpy as np
import sys

def prepare_input(board: Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

# Load the mapping

with open(r"C:\Users\Tuan Dung\IdeaProjects\CSCHTTT\.idea\Chess\Model2\models\heavy_move_to_int_1", "rb") as file:
    move_to_int = pickle.load(file)
num_classes = len(move_to_int)
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = ChessModel(num_classes=num_classes)
model.load_state_dict(torch.load(r"C:\Users\Tuan Dung\IdeaProjects\CSCHTTT\.idea\Chess\Model2\models\TORCH_final_model_100EPOCHS.pth"))
model.to(device)
model.eval()  # Set the model to evaluation mode (it may be reductant)

int_to_move = {v: k for k, v in move_to_int.items()}
# Function to make predictions
def predict_move(board: Board):
    X_tensor = prepare_input(board).to(device)

    with torch.no_grad():
        logits = model(X_tensor)

    logits = logits.squeeze(0)  # Remove batch dimension

    probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move

    return None

# Continuously read FEN from input
while True:
    try:
        fen = sys.stdin.readline().strip()  # Read each FEN from standard input
        if fen:  # If there's valid input, process it
            board = Board(fen)
            predicted_move = predict_move(board)
            print(predicted_move, flush=True)  # Print result and flush immediately for Unity to read
    except Exception as e:
        print(f"Error: {e}", flush=True)