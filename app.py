import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to create a Connect Four board
def create_board(rows, cols):
    return np.zeros((rows, cols), dtype=int)

# Function to drop a piece into the board
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Function to check if a column is valid for placing a piece
def is_valid_location(board, col):
    return board[0][col] == 0

# Function to get the next available row in a column
def get_next_open_row(board, col):
    for r in range(len(board)):
        if board[r][col] == 0:
            return r

# Function to print the board
def print_board(board):
    fig, ax = plt.subplots()
    ax.imshow(board, cmap='viridis')
    ax.axis('off')
    st.pyplot(fig)

# Streamlit app
def main():
    st.title('Connect Four')

    rows = 6
    cols = 7

    board = create_board(rows, cols)
    print_board(board)

    # Button to select columns
    col_buttons = []
    for col in range(cols):
        col_buttons.append(st.button(f'Column {col + 1}'))

    if any(col_buttons):
        col_index = col_buttons.index(True)
        st.write(f'Column {col_index + 1} clicked')

if __name__ == '__main__':
    main()
