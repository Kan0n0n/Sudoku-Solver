from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from sudoku_extract import preProccessing, findBiggestContour, wrapImage, splitCells, processCells, sudoku_extract, draw_solution_on_original, merge_images
from sudoku_solver import solve_sudoku
import base64
import tensorflow as tf
import os

app = Flask(__name__)

model_flask_path = os.path.join(app.static_folder, "model/num_classifier_150.tflite")
interpreter = tf.lite.Interpreter(model_path=model_flask_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        # Get image from POST request
        file = request.files['image']
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process image
        processed_img = preProccessing(frame)
        biggest_contour = findBiggestContour(processed_img)

        if biggest_contour is None:
            return jsonify({'error': 'No Sudoku puzzle found in image'})
        
        print('Found biggest contour')
        wrapped_img = wrapImage(frame, biggest_contour)
        cells = splitCells(wrapped_img)
        processed_cells = processCells(cells)
        
        sudoku_matrix = sudoku_extract(processed_cells, interpreter)
        original_matrix = sudoku_matrix.copy()

        print(sudoku_matrix)
        
        if not solve_sudoku(sudoku_matrix):
            return jsonify({'error': 'Could not solve Sudoku puzzle'})

        result_frame = draw_solution_on_original(frame, biggest_contour, original_matrix, sudoku_matrix - original_matrix)
        
        merge_frame = merge_images(frame, result_frame)

        _, buffer = cv2.imencode('.jpg', merge_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_str
        })

    except Exception as e:
        return jsonify({'error': 'No Sudoku puzzle found in image'})

if __name__ == '__main__':
    app.run(debug=True)
