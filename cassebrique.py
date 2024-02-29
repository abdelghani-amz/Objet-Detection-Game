import cv2 
import numpy as np
from random import  choice


def horizontalImpact(game, direction, ballx, bally):

    if (bally + direction[1]*8 >= 600) or (bally + direction[1]*(8) <= 0):
        direction[1] = direction[1] * (-1)

    elif np.any(game[ballx - 7: ballx + 8, bally + direction[1]*8] != 255):
        if np.all(game[ballx - 7: ballx + 8, bally + direction[1]*8] != 0):
            impactY = bally + direction[1]*8
            impactX = ballx

            brickX = impactX // 25
            brickY = impactY // 50

            if np.all(game[impactX, impactY] == 255): 
                brickX = (impactX + 8) // 25
                game[(brickX)*25 : (brickX + 1)*25, (brickY)*50 : (brickY + 1)*50,:] = 255
                brickX = (impactX - 8) // 25
                game[(brickX)*25 : (brickX + 1)*25, (brickY)*50 : (brickY + 1)*50,:] = 255
            else:
                game[(brickX)*25 : (brickX + 1)*25, (brickY)*50 : (brickY + 1)*50,:] = 255

        direction[1] = direction[1] * (-1)

    return direction


def verticalImpact(game, direction, ballx, bally):
    
    if (ballx + direction[0]*8 >= 400): #or (ballx + direction[0]*8 <= 0):
        direction = [0,0]
    elif (ballx + direction[0]*8 <= 0):
        direction[0] = direction[0] * (-1)
    
    elif np.any(game[ballx + direction[0]*8, bally - 7: bally + 8] != 255): 
        if np.all(game[ballx + direction[0]*8, bally - 7: bally + 8] != 0):
            impactY = bally 
            impactX = ballx + direction[0]*8
            brickX = impactX // 25
            brickY = impactY // 50

            if np.all(game[impactX, impactY] == 255):
                brickY = (impactY - 8) // 50
                game[(brickX)*25 : (brickX + 1)*25, (brickY)*50 : (brickY + 1)*50,:] = 255
                brickY = (impactY + 8) // 50
                game[(brickX)*25 : (brickX + 1)*25, (brickY)*50 : (brickY + 1)*50,:] = 255
            else:
                game[(brickX)*25 : (brickX + 1)*25, (brickY)*50 : (brickY + 1)*50,:] = 255

        direction[0] = direction[0] * (-1)

    return direction

def detect_inrange(image, lo, hi, surfMin, surfMax):
    frame = image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image = cv2.blur(image, (5,5))
    mask = cv2.inRange(image, lo, hi)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    circle = None
    try:
        elements = sorted(elements, key= lambda x: cv2.contourArea(x), reverse=True)
        for element in elements:
            if cv2.contourArea(element) >= surfMin and cv2.contourArea(element) <= surfMax:
                circle = cv2.minEnclosingCircle(element)[0]
                circle = int(circle[0]), int(circle[1])
                return frame, mask, circle
    except:
        pass

    return frame, mask, circle


def detectColor(VideoCap):
    while(True):
        ret, frame=VideoCap.read()
        cv2.flip(frame,1,frame)
        cv2.putText(frame ,"Put object in circle. Press key to continue", (0, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0,0,255), 1, cv2.LINE_AA )
        cv2.circle(frame, (299, 240), 30, (0, 255, 0), 5)
        cv2.imshow('image', frame)

        if cv2.waitKey(33)!=-1:
                break

    mean = []
    
    while True:
        ret, frame=VideoCap.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        cv2.flip(frame,1,frame)
        cv2.putText(frame ,"Give multiple viewing angles for around 5 seconds. Press key to continue", (0, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,255), 1, cv2.LINE_AA )
        cv2.circle(frame, (299, 240), 30, (0, 255, 0), 5)
        cv2.imshow('image', frame)

        mean.append(hsv[240, 299])
        if cv2.waitKey(33)!=-1:
            break
    
    cv2.destroyAllWindows()
    mean = np.array(mean)
    lo = np.array([np.min(mean[:, i]) for i in range(3)])
    hi = np.array([np.max(mean[:, i]) for i in range(3)])
    hi[1:3] = 255 - (255 - hi[1:3]) / 2

    return lo, hi


##### BRICKS #######
game = np.ones((400, 600, 3), np.uint8) * 255
game[50:100,50:550] = [255,1,1]
game[100:150,50:550] = [1,255,1]
game[150:200,50:550] = [1,1,255]
game[200:250,50:550] = [1,255,255]

for i in range(9):
    game[:, 100 + i*50 - 2 : 100 + i*50 + 3] = 255
for i in range(7):
    game[75 + i*25 - 2 : 75 + i*25 + 3, :] = 255


##### PLATEFORME #####
platW = 40
platX, platY = 381, 299
game[platX - 5: platX + 6, platY - platW: platY + 1 + platW, :] = 0
cv2.circle(game, (299, 340), 7, [0,0,0], thickness=-1)
directions = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
direction  = choice(directions)
ballx, bally = 340 , 299
ballMatrix = game[ballx-7 : ballx + 8, bally -7: bally +8].copy()
difficulty = 1

VideoCap=cv2.VideoCapture(0)
lo, hi = detectColor(VideoCap)
print("interval HSV : ", lo,hi)


while True:
    ret, frame=VideoCap.read()
    cv2.flip(frame,1,frame)

    image, mask, circle = detect_inrange(frame,lo, hi, 2500, 30000)
    if circle is not None:
        cv2.circle(image,circle, 10, (0,0,255), 5)
        game[platX - 5: platX + 6, platY - platW: platY + 1 + platW, :] = 255
        platY = circle[0]

        if platY + platW >= 600:
            platY = 599 - platW
        elif platY - platW <= 0:
            platY = platW

        game[platX - 5: platX + 6, platY - platW: platY + 1 + platW, :] = 0

    #print(image[100,100])
    
    for i in range(difficulty):
        game[ballx-7 : ballx + 8, bally -7: bally +8, :] = 255
        ballx = ballx + direction[0]
        bally = bally + direction[1]
        game[ballx - 7: ballx + 8, bally - 7: bally + 8] = ballMatrix

        ### IMPACT HORIZONTAL ####        
        direction = horizontalImpact(game, direction, ballx, bally)

        ### IMPACT VERTICAL ####
        direction = verticalImpact(game, direction, ballx, bally)

    cv2.imshow('image', image)
    cv2.imshow('mask', mask)
    cv2.imshow("game", game)

    key = cv2.waitKey(33)&0xFF
    if key == ord('w'):
        difficulty = difficulty + 1

    elif key == ord('s') and difficulty >= 1:
        difficulty = difficulty - 1

    elif key == ord('d') and platY + platW + difficulty*2 <= 600:
        game[platX - 5: platX + 6, platY - platW: platY + 1 + platW] = 255
        platY = platY + difficulty*2
        game[platX - 5: platX + 6, platY - platW: platY + 1 + platW] = 0
    
    elif key == ord('a') and platY - platW - difficulty*2 > 0:
        game[platX - 5: platX + 6, platY - platW: platY + 1 + platW] = 255
        platY = platY - difficulty*2
        game[platX - 5: platX + 6, platY - platW: platY + 1 + platW] = 0
    elif key == ord('z'):
        break





