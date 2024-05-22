import numpy as np
import cv2
import mediapipe as mp

# Load Model
hands = mp.solutions.hands
hand_landmark = hands.Hands(max_num_hands=1)

draw = mp.solutions.drawing_utils
# Camera frame resolution
frame_shape = (720, 1280, 3)

mask = np.zeros(frame_shape, dtype='uint8')
colour = (0, 0, 255)
thickness = 10
# Read toolbar image

tools = cv2.imread("tool.png")
tools = tools.astype('uint8')
# Row and Column for toolbar
midCol = 1280 // 2
max_row = 50
min_col = midCol-125
max_col = midCol+125
curr_tool = 'pencil'
start_point = None
cap = cv2.VideoCapture(0)
prevxy = None

# Check if distance between 2 points is less than 60 pixels


def get_is_clicked(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2

    dis = (x1-x2)**2 + (y1-y2)**2
    dis = np.sqrt(dis)
    if dis < 60:
        return True
    else:
        return False

# Return tool based on column location


def get_Tool(point, prev_tool):
    (x, y) = point

    if x > min_col and x < max_col and y < max_row:
        if x < min_col:
            return
        elif x < 50 + min_col:
            curr_tool = "line"
        elif x < 100 + min_col:
            curr_tool = "rectangle"
        elif x < 150 + min_col:
            curr_tool = "pencil"
        elif x < 200 + min_col:
            curr_tool = "circle"
        else:
            curr_tool = "erase"
        return curr_tool
    else:
        return prev_tool


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Preprocess Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    # Check if hand is in frame
    if op.multi_hand_landmarks:
        for all_landmarks in op.multi_hand_landmarks:
            draw.draw_landmarks(frame, all_landmarks, hands.HAND_CONNECTIONS)
            # index finger location
            x = all_landmarks.landmark[8].x * frame_shape[1]
            y = all_landmarks.landmark[8].y * frame_shape[0]
            x, y = int(x), int(y)

            # Middle finger location
            thumb_tip_x = all_landmarks.landmark[4].x * frame_shape[1]
            thumb_tip_y = all_landmarks.landmark[4].y * frame_shape[0]
            thumb_tip_x, thumb_tip_y = int(thumb_tip_x), int(thumb_tip_y)

            is_clicked = get_is_clicked((x, y), (thumb_tip_x, thumb_tip_y))
            curr_tool = get_Tool((x, y), curr_tool)

            cv2.putText(frame,
                        curr_tool,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0),
                        3,
                        cv2.LINE_4)

            # Select tool and draw for that
            if curr_tool == 'pencil':
                # Connect previous and current index finger locations
                if is_clicked and prevxy != None:
                    cv2.line(mask, prevxy, (x, y), colour, thickness)

            elif curr_tool == 'rectangle':
                if is_clicked and start_point == None:
                    # Init start_point
                    start_point = (x, y)
                elif is_clicked:
                    # Draw temp rectange
                    cv2.rectangle(frame, start_point,
                                  (x, y), colour, thickness)
                elif is_clicked == False and start_point:
                    # draw perm. rectangle and reset start_point
                    cv2.rectangle(mask, start_point, (x, y), colour, thickness)
                    start_point = None

            elif curr_tool == 'circle':
                if is_clicked and start_point == None:
                    start_point = (x, y)

                if start_point:
                    rad = int(
                        ((start_point[0]-x)**2 + (start_point[1]-y)**2)**0.5)
                if is_clicked:
                    cv2.circle(frame, start_point, rad, colour, thickness)

                if is_clicked == False and start_point:
                    cv2.circle(mask, start_point, rad, colour, thickness)

                    start_point = None

            elif curr_tool == "erase":
                cv2.circle(frame, (x, y), 30, (0, 0, 0), -1)  # -1 means fill
                if is_clicked:
                    cv2.circle(mask, (x, y), 30, 0, -1)
            prevxy = (x, y)

    # Merge Frame and Mask
    frame = np.where(mask, mask, frame)

    frame[0:max_row, min_col:max_col] = tools
    cv2.imshow('Live', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.waitKey(1)
