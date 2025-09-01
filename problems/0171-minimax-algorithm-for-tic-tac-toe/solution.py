import numpy as np

def is_winner(board, player):
    # Check rows, columns, and diagonals
    for i in range(3):
        if all(board[i, j] == player for j in range(3)):
            return True
        if all(board[j, i] == player for j in range(3)):
            return True
    if all(board[i, i] == player for i in range(3)):
        return True
    if all(board[i, 2-i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return not any(board[i, j] == '' for i in range(3) for j in range(3))

def get_available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == '']

def minimax_tictactoe(board: np.ndarray, player: str) -> tuple:
    """
    Returns the optimal move (row, col) for the given player ('X' or 'O') on the current board using Minimax.
    Args:
        board: 3x3 NumPy array with entries 'X', 'O', or ''
        player: 'X' or 'O'
    Returns:
        Tuple (row, col) for the optimal move
    """
    # Y