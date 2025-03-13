import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize Pygame mixer for sound
pygame.mixer.init()

# Set up MediaPipe hands detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Previous hand positions for swipe detection
prev_x_right = None
prev_y_right = None
swipe_threshold = 50  # Minimum movement in pixels to detect a swipe

# Load sounds for gestures and swipe actions
swipe_right_sound = pygame.mixer.Sound('C:/Users/hbeci/PycharmProjects/Thesis/swipe_right.mp3')
swipe_left_sound = pygame.mixer.Sound('C:/Users/hbeci/PycharmProjects/Thesis/swipe_left.mp3')
swipe_up_sound = pygame.mixer.Sound('C:/Users/hbeci/PycharmProjects/Thesis/swipe_up.mp3')
swipe_down_sound = pygame.mixer.Sound('C:/Users/hbeci/PycharmProjects/Thesis/swipe_down.mp3')

# Pinch gesture detection parameters
pinch_threshold = 30  # Distance threshold for pinch gesture
distance_threshold = 0.05  # Minimum change in distance to trigger spiral actions
previous_distance = None  # To track the distance between hands in the previous frame


def check_pinch(hand_landmarks):
    # Pinch is detected when the thumb tip and middle finger tip are very close
    thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
    middle_tip = hand_landmarks.landmark[12]  # Middle finger tip

    # Calculate Euclidean distance between thumb and middle finger tips
    distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y]))

    # Check if both fingers are sufficiently close (x, y axes) and distance is below threshold
    if distance < pinch_threshold and abs(thumb_tip.y - middle_tip.y) < 0.05:
        return True
    return False


def get_hand_center(hand_landmarks):
    # Calculate the center of the hand by averaging the x and y coordinates of all landmarks
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    return np.mean(x_coords), np.mean(y_coords)


def calculate_hand_distance(hand_landmarks_1, hand_landmarks_2):
    # Calculate the distance between the centers of two hands
    x1, y1 = get_hand_center(hand_landmarks_1)
    x2, y2 = get_hand_center(hand_landmarks_2)
    return np.linalg.norm([x1 - x2, y1 - y2])  # Euclidean distance between hand centers


def recognize_gesture(hand_landmarks, hand_label):
    fingers = []
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    for i in range(1, 5):  # Skip thumb for now
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    thumb = hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x

    if sum(fingers) == 0:
        # fist_sound.play()
        return "Fist"
    elif sum(fingers) == 4:
        # palm_sound.play()
        return "Palm"
    elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
        # peace_sound.play()
        return "Peace"


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for better user experience
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    left_hand_landmarks = None
    right_hand_landmarks = None
    left_hand_pinch = False
    right_hand_pinch = False

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            hand_x = np.mean(x_coords) * frame.shape[1]  # Convert to pixel coordinates
            hand_y = np.mean(y_coords) * frame.shape[0]  # Convert to pixel coordinates

            if hand_label == "Left":  # Gesture recognition for left hand
                left_hand_landmarks = hand_landmarks
                left_hand_pinch = check_pinch(left_hand_landmarks)
                if not left_hand_pinch:
                    gesture = recognize_gesture(hand_landmarks, hand_label)
                    cv2.putText(frame, f"Left Hand: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # print("Left Hand Gesture Detected:", gesture)

            elif hand_label == "Right":  # Swipe detection for right hand
                right_hand_landmarks = hand_landmarks
                right_hand_pinch = check_pinch(right_hand_landmarks)
                if not right_hand_pinch:
                    # Swipe detection for right hand
                    if prev_x_right is not None and prev_y_right is not None:
                        movement_x = hand_x - prev_x_right
                        movement_y = hand_y - prev_y_right

                        if movement_x > swipe_threshold:
                            print("Right Hand Swipe Right Detected")
                            swipe_right_sound.play()
                        elif movement_x < -swipe_threshold:
                            print("Right Hand Swipe Left Detected")
                            swipe_left_sound.play()

                        if movement_y > swipe_threshold:
                            print("Right Hand Swipe Down Detected")
                            swipe_down_sound.play()
                        elif movement_y < -swipe_threshold:
                            print("Right Hand Swipe Up Detected")
                            swipe_up_sound.play()

                    prev_x_right = hand_x
                    prev_y_right = hand_y

    if left_hand_landmarks and right_hand_landmarks:
        if left_hand_pinch and right_hand_pinch:
            # Both hands are pinched, calculate distance and check for spiral action
            current_distance = calculate_hand_distance(left_hand_landmarks, right_hand_landmarks)
            if previous_distance is not None:
                distance_change = abs(current_distance - previous_distance)
                if distance_change > distance_threshold:  # Significant movement detected
                    if current_distance < previous_distance:
                        print("Making a Left Spiral!")
                    elif current_distance > previous_distance:
                        print("Making a Right Spiral!")
            previous_distance = current_distance

    cv2.imshow("Hand Gesture & Swipe Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()