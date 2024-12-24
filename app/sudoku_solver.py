import numpy as np

def is_valid_move(matrix, row, col, num):
    if num in matrix[row]:
        return False
    if num in matrix[:, col]:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in matrix[start_row:start_row + 3, start_col:start_col + 3]:
        return False
    return True

def solve_sudoku(matrix):
    for row in range(9):
        for col in range(9):
            if matrix[row][col] == 0:
                for num in range(1, 10):
                    if is_valid_move(matrix, row, col, num):
                        matrix[row][col] = num
                        if solve_sudoku(matrix):
                            return True
                        matrix[row][col] = 0
                return False
    return True