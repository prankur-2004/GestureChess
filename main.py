import cv2
import chess
import numpy as np
import math
import sys
import os

if sys.platform.startswith('win'):
    os.environ["PYTHONIOENCODING"] = "utf-8"

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FRAME_SIZE = 800          
BOARD_MARGIN = 60         
BOARD_SIZE = FRAME_SIZE - 2 * BOARD_MARGIN
SQUARE = BOARD_SIZE // 8
PINCH_PIXEL_THRESHOLD = 35  
CAMERA_ID = 0

WHITE_SQ = (245, 245, 220)
BLACK_SQ = (139, 115, 85)
HIGHLIGHT = (255, 191, 0)
SELECT_COLOR = (0, 165, 255)
TEXT_COLOR = (40, 40, 40)

# Windows compatible standard display characters
PIECE_SYMBOLS = {
    'R': 'R', 'N': 'N', 'B': 'B', 'Q': 'Q', 'K': 'K', 'P': 'P',
    'r': 'r', 'n': 'n', 'b': 'b', 'q': 'q', 'k': 'k', 'p': 'p'
}

def pixel_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def finger_pos_to_square(px, py):
    left, top = BOARD_MARGIN, BOARD_MARGIN
    if px < left or px >= left + BOARD_SIZE or py < top or py >= top + BOARD_SIZE:
        return None
    file = int((px - left) // SQUARE)
    rank = 7 - int((py - top) // SQUARE)
    return chess.square(file, rank)

def draw_board(img, board, selected_square=None, hover_square=None, last_move=None):
    left, top = BOARD_MARGIN, BOARD_MARGIN
    for r in range(8):
        for f in range(8):
            x, y = left + f * SQUARE, top + (7 - r) * SQUARE 
            color = WHITE_SQ if (f + r) % 2 == 0 else BLACK_SQ
            cv2.rectangle(img, (x, y), (x + SQUARE, y + SQUARE), color, -1)

    if last_move is not None:
        for s in (last_move.from_square, last_move.to_square):
            f, r = chess.square_file(s), chess.square_rank(s)
            cv2.rectangle(img, (left + f * SQUARE, top + (7 - r) * SQUARE), (left + f * SQUARE + SQUARE, top + (7 - r) * SQUARE + SQUARE), (144, 238, 144), 3)

    if hover_square is not None:
        f, r = chess.square_file(hover_square), chess.square_rank(hover_square)
        cv2.rectangle(img, (left + f * SQUARE, top + (7 - r) * SQUARE), (left + f * SQUARE + SQUARE, top + (7 - r) * SQUARE + SQUARE), HIGHLIGHT, 3)

    if selected_square is not None:
        f, r = chess.square_file(selected_square), chess.square_rank(selected_square)
        cv2.rectangle(img, (left + f * SQUARE, top + (7 - r) * SQUARE), (left + f * SQUARE + SQUARE, top + (7 - r) * SQUARE + SQUARE), SELECT_COLOR, -1)

    for sq, piece in board.piece_map().items():
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x, y = left + f * SQUARE, top + (7 - r) * SQUARE
        symbol = PIECE_SYMBOLS.get(piece.symbol(), piece.symbol())
        cv2.putText(img, symbol, (x + SQUARE // 2 - 15, y + SQUARE // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2, cv2.LINE_AA)

    cv2.rectangle(img, (left, top), (left + BOARD_SIZE, top + BOARD_SIZE), (60, 60, 60), 3)

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE)

    board = chess.Board()
    selected_sq = None
    grabbed = False
    last_move = None
    pinch_frames = 0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(cv2.flip(frame, 1), (FRAME_SIZE, FRAME_SIZE))
            canvas = frame.copy()
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            index_tip_px, raw_pinch = None, False

            if results.multi_hand_landmarks:
                h_lm = results.multi_hand_landmarks[0].landmark
                index_tip_px = (int(h_lm[8].x * FRAME_SIZE), int(h_lm[8].y * FRAME_SIZE))
                thumb_tip_px = (int(h_lm[4].x * FRAME_SIZE), int(h_lm[4].y * FRAME_SIZE))
                if pixel_dist(index_tip_px, thumb_tip_px) < PINCH_PIXEL_THRESHOLD:
                    raw_pinch = True
                mp_drawing.draw_landmarks(canvas, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            pinch_frames = pinch_frames + 1 if raw_pinch else max(0, pinch_frames - 1)
            pinch = pinch_frames >= 3

            hover_sq = finger_pos_to_square(index_tip_px[0], index_tip_px[1]) if index_tip_px else None

            if pinch and not grabbed and hover_sq is not None:
                piece = board.piece_at(hover_sq)
                if piece and piece.color == board.turn:
                    selected_sq, grabbed = hover_sq, True
            elif grabbed and not pinch:
                if selected_sq is not None and hover_sq is not None and selected_sq != hover_sq:
                    move = chess.Move(selected_sq, hover_sq)
                    if move not in board.legal_moves:
                        move_q = chess.Move(selected_sq, hover_sq, promotion=chess.QUEEN)
                        move = move_q if move_q in board.legal_moves else None
                    if move in board.legal_moves:
                        board.push(move)
                        last_move = board.peek()
                selected_sq, grabbed = None, False

            overlay = np.ones_like(canvas) * 255
            draw_board(overlay, board, selected_sq, hover_sq, last_move)
            blended = cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0)

            if index_tip_px:
                col = SELECT_COLOR if grabbed else HIGHLIGHT
                cv2.circle(blended, index_tip_px, 10, col, -1)
                cv2.putText(blended, "GRABBED" if grabbed else "HOVER", (index_tip_px[0] + 15, index_tip_px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            cv2.imshow("Gesture Chess", blended)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('u') and board.move_stack:
                board.pop()
                last_move = board.peek() if board.move_stack else None
            elif key == ord('r'):
                board.reset()
                selected_sq, grabbed, last_move = None, False, None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
