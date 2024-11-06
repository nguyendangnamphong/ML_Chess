import os
import numpy as np
from tqdm import tqdm
from chess import pgn
from auxiliary_func import board_to_matrix

def build_move_dict(moves):
    move_dict = {move: idx for idx, move in enumerate(sorted(set(moves)))}
    return move_dict

def load_pgn(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def process_and_save_games_in_chunks(files, chunk_size=10000, save_dir="D:\DevGame\ChessDataSet"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    X, y = [], []
    chunk_index = 0
    all_moves = set()  # To collect all possible moves

    for file in tqdm(files[:41], desc="Processing files"):
        for game in load_pgn(rf"C:\Users\Tuan Dung\IdeaProjects\CSCHTTT\.idea\Chess\LichessEliteDatabase\{file}"):
            board = game.board()
            for move in game.mainline_moves():
                X.append(board_to_matrix(board))  # Convert board to features
                y.append(move.uci())              # Collect move as a string initially
                all_moves.add(move.uci())
                board.push(move)

                if len(X) >= chunk_size:
                    # Encode moves after collecting a chunk
                    move_dict = build_move_dict(all_moves)
                    y_encoded = [move_dict[move] for move in y]  # Convert y to integer encoding

                    # Save the current chunk
                    np.save(f"{save_dir}/X_chunk_{chunk_index}.npy", np.array(X, dtype=np.float32))
                    np.save(f"{save_dir}/y_chunk_{chunk_index}.npy", np.array(y_encoded, dtype=np.int64))

                    X, y = [], []  # Clear lists for the next chunk
                    chunk_index += 1

    # Handle any leftover data after the loop
    if X:
        move_dict = build_move_dict(all_moves)
        y_encoded = [move_dict[move] for move in y]
        np.save(f"{save_dir}/X_chunk_{chunk_index}.npy", np.array(X, dtype=np.float32))
        np.save(f"{save_dir}/y_chunk_{chunk_index}.npy", np.array(y_encoded, dtype=np.int64))

    # Save move dictionary for use during model training and inference
    np.save(f"{save_dir}/move_dict.npy", move_dict)
    print(f"Data saved in {chunk_index + 1} chunks.")



'''# Create move_dict with all moves
move_dict = build_move_dict(all_moves)

process_and_save_games_in_chunks(files)'''
