import cv2
import numpy as np
import serial

def pick_up_ball(serial_port):
    ser = serial.Serial(serial_port, 9600, timeout=1)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, img = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([107, 101, 91])
        upper_blue = np.array([130, 216, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(img, img, mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            ser.write(f'{center_x},{center_y}\n'.encode())

        cv2.imshow('img', img)
        cv2.imshow('mask', mask)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    pick_up_ball('/dev/ttyACM0')
