import numpy as np
from chess import Board
import os
import torch


def board_to_matrix(board: Board):
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    # maybe 14th for squares FROM WHICH we can move? idk
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix


def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)

def get_input_y(games):
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            y.append(move.uci())
            board.push(move)
    return np.array(y)

def create_input_for_nn_in_chunks(games, chunk_size=100000, output_dir="D:\DevGame\ChessDataSet"):
    X_chunk = []
    y_chunk = []
    chunk_index = 0

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X_chunk.append(board_to_matrix(board))
            y_chunk.append(move.uci())
            board.push(move)

            # Check if chunk size is reached
            if len(X_chunk) >= chunk_size:
                # Save the chunk to disk
                X_chunk_np = np.array(X_chunk, dtype=np.float32)
                y_chunk_np = np.array(y_chunk)

                torch.save((X_chunk_np, y_chunk_np), f"{output_dir}/chunk_{chunk_index}.pt")
                print(f"Saved chunk {chunk_index} with {len(X_chunk)} samples")

                # Clear the lists and increment the chunk index
                X_chunk.clear()
                y_chunk.clear()
                chunk_index += 1

    # Save any remaining data as the last chunk
    if X_chunk:
        X_chunk_np = np.array(X_chunk, dtype=np.float32)
        y_chunk_np = np.array(y_chunk)
        torch.save((X_chunk_np, y_chunk_np), f"{output_dir}/chunk_{chunk_index}.pt")
        print(f"Saved chunk {chunk_index} with {len(X_chunk)} samples")

def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int