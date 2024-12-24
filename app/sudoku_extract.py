import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



def preProccessing(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    _, thresh_image = cv2.threshold(blur_gray_image, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    dilate_image = cv2.dilate(thresh_image, kernel, iterations=1)
    return dilate_image

def is_valid_sudoku_contour(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) != 4:
        return False
    
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    side_lengths = [
        distance(rect[0], rect[1]),
        distance(rect[1], rect[2]),
        distance(rect[2], rect[3]),
        distance(rect[3], rect[0])
    ]
    
    avg_side_length = np.mean(side_lengths)
    if not all(abs(side - avg_side_length) / avg_side_length < 0.1 for side in side_lengths):
        return False
    
    def angle(p1, p2, p3):
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ab = b - a
        bc = b - c
        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    angles = [
        angle(rect[0], rect[1], rect[2]),
        angle(rect[1], rect[2], rect[3]),
        angle(rect[2], rect[3], rect[0]),
        angle(rect[3], rect[0], rect[1])
    ]
    
    if not all(80 <= angle <= 100 for angle in angles):
        return False
    
    return True

def findBiggestContour(preProccessed_img):
    contours, _ = cv2.findContours(preProccessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    if not is_valid_sudoku_contour(biggest_contour):
        return None
    return biggest_contour

def drawBiggestContour(img, biggest_contour):
    cv2.drawContours(img, [biggest_contour], -1, (0, 255, 0), 3)
    return img

def wrapImage(img, biggest_contour):
    epsilon = 0.02 * cv2.arcLength(biggest_contour, True)
    approx = cv2.approxPolyDP(biggest_contour, epsilon, True)

    pts = approx.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    width = 450
    height = 450
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    wrap = cv2.warpPerspective(img, M, (width, height))

    return wrap

def splitCells(wrap):
    rows = np.vsplit(wrap, 9)
    cells = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for cell in cols:
            cells.append(cell)
    return cells

def processCells(cells):
    MIN_SIZE = 450
    MAX_SIZE = 22500
    EDGE_THRESHOLD = 3
    w, h = 150, 150

    processed_cells = []

    for i in range(len(cells)):
        target_image = cells[i]

        gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.resize(gray_target, (w, h))

        _, binary_image = cv2.threshold(gray_target, 180, 255, cv2.THRESH_BINARY_INV)

        num_labels, labels_im = cv2.connectedComponents(binary_image)

        best_component = None
        best_size = 0
        height, width = gray_target.shape
        center_y, center_x = height // 2, width // 2

        for label in range(1, num_labels):
            component_mask = (labels_im == label).astype(np.uint8)
            size = np.sum(component_mask)

            y_indices, x_indices = np.where(component_mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            min_y, max_y = np.min(y_indices), np.max(y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)

            if (MIN_SIZE <= size <= MAX_SIZE and
                min_y > EDGE_THRESHOLD and max_y < height - EDGE_THRESHOLD and
                min_x > EDGE_THRESHOLD and max_x < width - EDGE_THRESHOLD):

                if size > best_size:
                    best_size = size
                    best_component = component_mask

        if best_component is not None:
            mask = best_component.astype(np.uint8) * 255

            output_image = np.zeros_like(gray_target)
            output_image[mask > 0] = 255

            processed_cells.append(output_image)
        else:
            processed_cells.append(np.zeros_like(gray_target))

    return processed_cells

def cell_preprocessing(cell):
    w = 150
    h = 150
    cell = cv2.resize(cell, (w, h))
    cell = cv2.equalizeHist(cell)
    _, binary_image = cv2.threshold(cell, 180, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.reshape(w, h, 1)
    binary_image = binary_image / 255.0
    return binary_image

def sudoku_extract(cells, interpreter):
    sudoku_matrix = np.zeros((9, 9), dtype=int)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(9):
        for j in range(9):
            index = i * 9 + j
            if index < len(cells):
                if np.sum(cells[index]) > 0:
                    test_cell = cells[index]
                    test_cell_preprocessed = cell_preprocessing(test_cell)
                    test_cell_preprocessed = np.expand_dims(test_cell_preprocessed, axis=0)

                    interpreter.set_tensor(input_details[0]['index'], test_cell_preprocessed.astype(np.float32))

                    interpreter.invoke()

                    y_pred = interpreter.get_tensor(output_details[0]['index'])
                    y_pred_labels = np.argmax(y_pred, axis=1)
                    sudoku_matrix[i][j] = y_pred_labels[0]
                else:
                    sudoku_matrix[i][j] = 0

    return sudoku_matrix

def draw_solution_on_original(original_img, biggest_contour, original_numbers, solved_numbers):
    original_img = original_img.copy()
    blank_image = np.zeros_like(original_img)
    approx = cv2.approxPolyDP(biggest_contour, 0.02 * cv2.arcLength(biggest_contour, True), True)
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    width, height = 450, 450
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(blank_image, M, (width, height))
    cell_width = warp.shape[1] // 9
    cell_height = warp.shape[0] // 9
    for i in range(9):
        for j in range(9):
            if original_numbers[i][j] != 0:
                continue
            x = j * cell_width
            y = i * cell_height
            cv2.putText(warp, str(solved_numbers[i][j]), (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    inv_M = cv2.getPerspectiveTransform(dst, rect)
    inv_warp = cv2.warpPerspective(warp, inv_M, (original_img.shape[1], original_img.shape[0]))
    inv_mask = cv2.bitwise_not(inv_warp)
    result = cv2.bitwise_and(original_img, inv_mask)
    #result = cv2.addWeighted(inv_warp, 1, original_img, 0.5, 1)
    return result

def merge_images(original_img, result_image):
    height, width = original_img.shape[0], original_img.shape[1]
    result_image = cv2.resize(result_image, (width, height))

    arrow_width = 50
    label_height = 70
    canvas_width = width * 2 + arrow_width
    canvas_height = height + label_height
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    background_color = original_img[0, 0].tolist()
    canvas[:label_height, :] = background_color
    canvas[:, width:width + arrow_width]  = background_color

    canvas[label_height:, :width] = original_img

    arrow_start = (width + 10, height // 2)
    arrow_end = (width + arrow_width - 10, height // 2)
    cv2.arrowedLine(canvas, arrow_start, arrow_end, (0, 0, 0), 5, tipLength=0.5)

    canvas[label_height:, width + arrow_width:] = result_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    text_color = (0, 0, 0)
    
    original_text = 'Original Image'
    solved_text = 'Solved Image'
    (orig_text_width, orig_text_height), _ = cv2.getTextSize(original_text, font, font_scale, font_thickness)
    (solved_text_width, solved_text_height), _ = cv2.getTextSize(solved_text, font, font_scale, font_thickness)
    
    orig_text_x = (width - orig_text_width) // 2
    solved_text_x = width + arrow_width + (width - solved_text_width) // 2
    
    cv2.putText(canvas, original_text, (orig_text_x, label_height - 20), font, font_scale, text_color, font_thickness)
    cv2.putText(canvas, solved_text, (solved_text_x, label_height - 20), font, font_scale, text_color, font_thickness)

    return canvas