import cv2
import numpy as np

# HSV – RGB – YUV

def Impresion(namme,imagen,x,y):
    cv2.namedWindow(namme)
    cv2.moveWindow(namme, x,y)
    cv2.imshow(namme, imagen)
    
'''def Rango(imagen,lower,upper):
    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(imagen, lower, upper)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    return (mask,res)'''
    
cap = cv2.VideoCapture('RAfrodita.mp4')

#Trnsformaciones Lineales.
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0,150,50])
        upper = np.array([255,255,180])

        mask = cv2.inRange(hvs, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask = mask)

        kernel = np.ones((15,15),np.float32)/225
        
        smoothed = cv2.filter2D(res,-1,kernel)
        blur = cv2.GaussianBlur(res,(15,15),0)
        median = cv2.medianBlur(res,15)
        bilateral = cv2.bilateralFilter(res,15,75,75)

        Impresion('frame',frame,50,10)
        Impresion('mask',mask,550,10)
        Impresion('res',res,1050,10)
        Impresion('smoothed',smoothed,550,270)
        Impresion('blur', blur,1050,270)
        Impresion('median', median,550,550)
        Impresion('bilateral', bilateral,1050,550)

        if cv2.waitKey(30) == ord('s'):
            break
    else: break

cv2.destroyAllWindows()
#cap.release()

#Treasformaciones Morfologicas
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0,150,50])
        upper = np.array([255,255,180])

        mask = cv2.inRange(hvs, lower, upper)
        
        res = cv2.bitwise_and(frame, frame, mask = mask)

        kernel = np.ones((5, 5), np.uint8)
        
        erosion = cv2.erode(res, kernel, iterations=1)
        dilation = cv2.dilate(res, kernel, iterations=1)
        fp = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        fn = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

        Impresion('frame',frame,50,10)
        Impresion('mask',mask,550,10)
        Impresion('res',res,1050,10)
        Impresion('erosion',erosion,550,270)
        Impresion('dilation', dilation,1050,270)
        Impresion('open', fp,550,550)
        Impresion('close', fn,1050,550)

        if cv2.waitKey(30) == ord('s'):
            break
    else: break

cv2.destroyAllWindows()
cap.release()



