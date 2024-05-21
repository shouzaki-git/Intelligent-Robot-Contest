import cv2
import numpy as np
import math
PI = math.pi


import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(0)

#解像度の設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

#焦点合わせ
focus_val = 125  # 0～255、5刻みの数字しか受け付けない
cap.set(cv2.CAP_PROP_FOCUS, focus_val)

#def undistortion_f(img):
#    cam_mtx = np.loadtxt("mtx.csv", dtype = "float", delimiter = ",")
#    distortion = np.loadtxt("d.csv", dtype = "float", delimiter = ",")
#    h, w = img.shape[:2]
#    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx,distortion,(w,h),1,(w,h))
#    dst = cv2.undistort(img, cam_mtx, distortion)
#    x,y,w,h = roi
#    dst = dst[y:y+h, x:x+w]
#    return dst



def distance_calculate(x,y):
   #coordinate transition
     X = x - 640
     Y = 360 - y
     #change is needed
     f = 737.257192
     thetac = PI/3
     H = 800#mm
     thetay = math.atan(Y/f)
     Wy = H/(math.tan(thetac+thetay))
     fy =math.sqrt(Y**2 + X**2)
     Wx = X*math.sqrt(H**2+Wy**2)/fy
     return Wx,Wy

def nothing(x):
        pass
cv2.namedWindow('Edge')
cv2.createTrackbar("threshold1","Edge", 189, 500, nothing)
cv2.createTrackbar("threshold2","Edge", 127, 500, nothing)

cv2.namedWindow('trackbar_Hough')
cv2.createTrackbar("dp", "trackbar_Hough", 12, 20, nothing)
cv2.createTrackbar("param1", "trackbar_Hough", 49, 120, nothing)
cv2.createTrackbar("param2", "trackbar_Hough", 28, 100, nothing)

cv2.namedWindow('trackbar_Hough_Y')
cv2.createTrackbar("dp", "trackbar_Hough_Y", 8, 20, nothing)
cv2.createTrackbar("param1", "trackbar_Hough_Y", 90, 120, nothing)
cv2.createTrackbar("param2", "trackbar_Hough_Y", 27, 100, nothing)

cv2.namedWindow('HSV_Blue')
cv2.createTrackbar("H_l_B", "HSV_Blue", 30, 180, nothing)
cv2.createTrackbar("S_l_B", "HSV_Blue", 181, 255, nothing)
cv2.createTrackbar("V_l_B", "HSV_Blue", 154, 255, nothing)
cv2.createTrackbar("H_h_B", "HSV_Blue", 133, 180, nothing)
cv2.createTrackbar("S_h_B", "HSV_Blue", 255, 255, nothing)
cv2.createTrackbar("V_h_B", "HSV_Blue", 255, 255, nothing)

cv2.namedWindow('HSV_Yellow')
cv2.createTrackbar("H_l_Y", "HSV_Yellow", 15, 180, nothing)
cv2.createTrackbar("S_l_Y", "HSV_Yellow", 86, 255, nothing)
cv2.createTrackbar("V_l_Y", "HSV_Yellow", 141, 255, nothing)
cv2.createTrackbar("H_h_Y", "HSV_Yellow", 50, 180, nothing)
cv2.createTrackbar("S_h_Y", "HSV_Yellow", 255, 255, nothing)
cv2.createTrackbar("V_h_Y", "HSV_Yellow", 255, 255, nothing)

cv2.namedWindow('HSV_Red1')
cv2.createTrackbar("H_l_R1", "HSV_Red1", 132, 180, nothing)
cv2.createTrackbar("S_l_R1", "HSV_Red1", 116, 255, nothing)
cv2.createTrackbar("V_l_R1", "HSV_Red1", 137, 255, nothing)
cv2.createTrackbar("H_h_R1", "HSV_Red1", 180, 180, nothing)
cv2.createTrackbar("S_h_R1", "HSV_Red1", 255, 255, nothing)
cv2.createTrackbar("V_h_R1", "HSV_Red1", 255, 255, nothing)

cv2.namedWindow('HSV_Red2')
cv2.createTrackbar("H_l_R2", "HSV_Red2", 117, 180, nothing)
cv2.createTrackbar("S_l_R2", "HSV_Red2", 146, 255, nothing)
cv2.createTrackbar("V_l_R2", "HSV_Red2", 149, 255, nothing)
cv2.createTrackbar("H_h_R2", "HSV_Red2", 180, 180, nothing)
cv2.createTrackbar("S_h_R2", "HSV_Red2", 255, 255, nothing)
cv2.createTrackbar("V_h_R2", "HSV_Red2", 255, 255, nothing)

while True:
    #usb cam 読み込み
    ret, frame = cap.read()
    if not ret: break

    #frameをhsv変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Edge
    threshold1 = cv2.getTrackbarPos("threshold1","Edge")
    threshold2 = cv2.getTrackbarPos("threshold2","Edge")

    #Hough
    Dp = cv2.getTrackbarPos("dp", "trackbar_Hough")
    Param1 = cv2.getTrackbarPos("param1", "trackbar_Hough")
    Param2 = cv2.getTrackbarPos("param2", "trackbar_Hough")

    if Dp > 0:
        Dp /= 10
    else:
        Dp = 0.1
    if Param1 == 0:
        Param1 = 0.1
    if Param2 == 0:
        Param2 = 0.1
    Dp_Y = cv2.getTrackbarPos("dp", "trackbar_Hough_Y")
    Param1_Y = cv2.getTrackbarPos("param1", "trackbar_Hough_Y")
    Param2_Y = cv2.getTrackbarPos("param2", "trackbar_Hough_Y")
    if Dp_Y > 0:
        Dp_Y /= 10
    else:
        Dp_Y = 0.1
    if Param1_Y == 0:
        Param1_Y = 0.1
    if Param2_Y == 0:
        Param2_Y = 0.1

    #HSV値の取得、入力
    h_l_B = cv2.getTrackbarPos("H_l_B", "HSV_Blue")
    h_h_B = cv2.getTrackbarPos("H_h_B", "HSV_Blue")
    s_l_B = cv2.getTrackbarPos("S_l_B", "HSV_Blue")
    s_h_B = cv2.getTrackbarPos("S_h_B", "HSV_Blue")
    v_l_B = cv2.getTrackbarPos("V_l_B", "HSV_Blue")
    v_h_B = cv2.getTrackbarPos("V_h_B", "HSV_Blue")

    h_l_Y = cv2.getTrackbarPos("H_l_Y", "HSV_Yellow")
    h_h_Y = cv2.getTrackbarPos("H_h_Y", "HSV_Yellow")
    s_l_Y = cv2.getTrackbarPos("S_l_Y", "HSV_Yellow")
    s_h_Y = cv2.getTrackbarPos("S_h_Y", "HSV_Yellow")
    v_l_Y = cv2.getTrackbarPos("V_l_Y", "HSV_Yellow")
    v_h_Y = cv2.getTrackbarPos("V_h_Y", "HSV_Yellow")

    h_l_R1 = cv2.getTrackbarPos("H_l_R1", "HSV_Red1")
    h_h_R1 = cv2.getTrackbarPos("H_h_R1", "HSV_Red1")
    s_l_R1 = cv2.getTrackbarPos("S_l_R1", "HSV_Red1")
    s_h_R1 = cv2.getTrackbarPos("S_h_R1", "HSV_Red1")
    v_l_R1 = cv2.getTrackbarPos("V_l_R1", "HSV_Red1")
    v_h_R1 = cv2.getTrackbarPos("V_h_R1", "HSV_Red1")

    h_l_R2 = cv2.getTrackbarPos("H_l_R2", "HSV_Red2")
    h_h_R2 = cv2.getTrackbarPos("H_h_R2", "HSV_Red2")
    s_l_R2 = cv2.getTrackbarPos("S_l_R2", "HSV_Red2")
    s_h_R2 = cv2.getTrackbarPos("S_h_R2", "HSV_Red2")
    v_l_R2 = cv2.getTrackbarPos("V_l_R2", "HSV_Red2")
    v_h_R2 = cv2.getTrackbarPos("V_h_R2", "HSV_Red2")

    yel_lower = np.array([h_l_Y, s_l_Y, v_l_Y])
    yel_upper = np.array([h_h_Y, s_h_Y, v_h_Y])

    blu_lower = np.array([h_l_B, s_l_B, v_l_B])
    blu_upper = np.array([h_h_B, s_h_B, v_h_B])

    red_lower1 = np.array([h_l_R1, s_l_R1, v_l_R1])
    red_upper1 = np.array([h_h_R1, s_h_R1, v_h_R1])

    red_lower2 = np.array([h_l_R2, s_l_R2, v_l_R2])
    red_upper2 = np.array([h_h_R2, s_h_R2, v_h_R2])

    #mask image
    img_mask_yellow = cv2.inRange(hsv, yel_lower, yel_upper)
    img_mask_blue = cv2.inRange(hsv, blu_lower, blu_upper)
    img_mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    img_mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    img_mask_red = img_mask_red1 | img_mask_red2

    img_mask = img_mask_yellow | img_mask_blue | img_mask_red

    #masking
    img_yellow = cv2.bitwise_and(frame ,frame, mask=img_mask_yellow)
    img_blue = cv2.bitwise_and(frame ,frame, mask=img_mask_blue)
    img_red = cv2.bitwise_and(frame ,frame, mask=img_mask_red)
    img_color = cv2.bitwise_and(frame ,frame, mask=img_mask)
    kernel=np.array([[0,0,1,1,1,0,0],
                        [0,1,1,1,1,1,0],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1],
                        [0,1,1,1,1,1,0],
                        [0,0,1,1,1,0,0]],dtype=np.uint8)
    kernel_y_f=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    kernel_y_s=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    #molphology

    #エッジ保存のため
    blur_yellow = cv2.bilateralFilter(img_yellow,20,0,0)
    edges = cv2.Canny(blur_yellow,threshold1,threshold2)
    edges = cv2.bitwise_not(edges)
    
    blur_yellow = cv2.bitwise_and(blur_yellow ,blur_yellow, mask=edges)
    erosion_yellow = cv2.erode(blur_yellow,kernel,iterations = 1)
    dilation_yellow = cv2.dilate(erosion_yellow,kernel,iterations = 1)
    blur_yellow = cv2.bilateralFilter(dilation_yellow,10,0,0)
    blur_yellow = cv2.bitwise_and(blur_yellow ,blur_yellow, mask=edges)

    blur_blue = cv2.GaussianBlur(img_blue,(7,7),0)
    erosion_blue = cv2.erode(blur_blue,kernel,iterations = 1)
    dilation_blue = cv2.dilate(erosion_blue,kernel,iterations = 1)
    blur_blue = cv2.GaussianBlur(dilation_blue,(7,7),0)

    blur_red = cv2.GaussianBlur(img_red,(7,7),0)
    erosion_red = cv2.erode(blur_red,kernel,iterations = 1)
    dilation_red = cv2.dilate(erosion_red,kernel,iterations = 1)
    blur_red = cv2.GaussianBlur(dilation_red,(7,7),0)

    gray_yellow = cv2.cvtColor(blur_yellow,cv2.COLOR_BGR2GRAY)
    gray_blue = cv2.cvtColor(blur_blue,cv2.COLOR_BGR2GRAY)
    gray_red = cv2.cvtColor(blur_red,cv2.COLOR_BGR2GRAY)
    gray = gray_yellow|gray_blue|gray_red
    
    #黄色ボールの円検出
    circles_yellow = cv2.HoughCircles(gray_yellow, cv2.HOUGH_GRADIENT,dp=Dp_Y,minDist = 20,param1=Param1_Y,param2=Param2_Y,minRadius=10,maxRadius=0)
    if circles_yellow is not None:
        circles_yellow = np.uint16(np.around(circles_yellow))
        for l in circles_yellow[0, :]:
            cv2.circle(frame,(l[0],l[1]),l[2],(0,255,255),2)
            cv2.circle(frame,(l[0],l[1]),2,(0,0,255),3)
            yx,yy = distance_calculate(l[0],l[1])
            text = str([int(yx), int(yy)])
            cv2.putText(frame,text,org=(l[0],l[1]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 255, 0),thickness=2)
            print("1:")
            print(yx,yy)


    #青色ボールの円検出
    circles_blue = cv2.HoughCircles(gray_blue,cv2.HOUGH_GRADIENT,dp= Dp,minDist = 20,param1=Param1,param2=Param2,minRadius=10,maxRadius=0)
    if circles_blue is not None:
        circles_blue = np.uint16(np.around(circles_blue))
        for n in circles_blue[0, :]:
            cv2.circle(frame,(n[0],n[1]),n[2],(255,0,0),2)
            cv2.circle(frame,(n[0],n[1]),2,(0,0,255),3)
            bx,by = distance_calculate(n[0],n[1])
            print("2:")
            print(bx,by)
            text = str([int(bx), int(by)])
            cv2.putText(frame,text,org=(n[0],n[1]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 255, 0),thickness=2)

    #赤色ボールの円検出
    circles_red = cv2.HoughCircles(gray_red,cv2.HOUGH_GRADIENT,dp= Dp,minDist = 20,param1=Param1,param2=Param2,minRadius=10,maxRadius=0)
    if circles_red is not None:
        circles_red = np.uint16(np.around(circles_red))
        for m in circles_red[0, :]:    
            cv2.circle(frame,(m[0],m[1]),m[2],(0,0,255),2)
            cv2.circle(frame,(m[0],m[1]),2,(0,0,255),3)
            rx,ry = distance_calculate(m[0],m[1])
            print("3:")
            print(rx,ry)
            text = str([int(rx), int(ry)])
            cv2.putText(frame,text,org=(m[0],m[1]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 255, 0),thickness=2)

    #cv2.imshow("circles", frame)
        #Edge
    threshold1 = cv2.getTrackbarPos("threshold1","Edge")
    threshold2 = cv2.getTrackbarPos("threshold2","Edge")
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWildows()

