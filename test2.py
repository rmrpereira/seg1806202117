import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt

numero_control = [0, 0]
numero_total = [0]

numero_controlred = [0, 0]
numero_red = [0]

numero_controlyellow = [0, 0]
numero_yellow = [0]

numero_controlgreen = [0, 0]
numero_green = [0]

numero_control4x2 = [0, 0]
numero_4x2 = [0]

numero_control2x2 = [0, 0]
numero_2x2 = [0]

numero_control4x2red = [0, 0]
numero_4x2red = [0]

numero_control4x2yellow = [0, 0]
numero_4x2yellow = [0]

numero_control4x2green = [0, 0]
numero_4x2green = [0]

numero_control2x2red = [0, 0]
numero_2x2red = [0]

numero_control2x2yellow = [0, 0]
numero_2x2yellow = [0]

numero_control2x2green = [0, 0]
numero_2x2green = [0]

def camara():
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    ret, frame = cap.read()
    cv.imwrite("teste.jpeg",frame)
    cap.release()
    
def color_range():
    aux = []
    name = []
    color = []
    f = open('Red.txt', 'r')
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    f = open('Green.txt', 'r')
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    f = open('Yellow.txt', 'r')
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    n = f.readline()
    n = np.uint8(n)
    aux.append(n)
    
    f.close()

    color = [[aux[0], aux[1], aux[2]], [aux[3], aux[4], aux[5]],
             [aux[6], aux[7], aux[8]], [aux[9], aux[10], aux[11]],
             [aux[12], aux[13], aux[14]], [aux[15], aux[16], aux[17]]]#,
             #[aux[18], aux[19], aux[20]], [aux[21], aux[22], aux[23]]#,
             #[aux[24], aux[25], aux[26]], [aux[27], aux[28], aux[29]],
             #[aux[30], aux[31], aux[32]], [aux[33], aux[34], aux[35]],
             #[aux[36], aux[37], aux[38]], [aux[39], aux[40], aux[41]],
             #[aux[42], aux[43], aux[44]], [aux[45], aux[46], aux[47]]]

    name = ['Red','Green','Yellow']
    #name = ['Red','Green','Blue','Yellow','Brown','Orange','Violet','Black']
    return color,name
#colortable
def filterblur(image): #filtro blur

    blur = cv.blur(image, (5, 5))
    blur = cv.bilateralFilter(blur, 9, 75, 75)
     
    return blur

def MostrarImage(img,imagename):    #Mostra a imagem
    newimage = cv.resize(img,None, fx=1,fy=1) 
    cv.imshow(imagename,newimage)

def mask_funct(hsvFrame,l,u):
    lower = np.array(l,np.uint8)
    upper = np.array(u,np.uint8)
    mask = cv.inRange(hsvFrame,lower,upper)
    return mask
def masks(hsvFrame):
    color,array_mask = color_range()

    for k in range(int(len(array_mask))):
        array_mask[k] = mask_funct(hsvFrame,color[2*k],color[2*k+1])
    return array_mask

def numerototal(control):
    numero_control[0] = numero_control[1]    
    numero_control[1] = control
    if(control > numero_control[0] or numero_total[0]==0):
        numero_total[0] = numero_total[0] + (control-numero_control[0])

def numerototalred(control_red):
    numero_controlred[0] = numero_controlred[1]    
    numero_controlred[1] = control_red
    if(control_red > numero_controlred[0] or numero_red[0]==0):
        numero_red[0] = numero_red[0] + (control_red-numero_controlred[0])

def numerototalyellow(control_yellow):
    numero_controlyellow[0] = numero_controlyellow[1]    
    numero_controlyellow[1] = control_yellow
    if(control_yellow > numero_controlyellow[0] or numero_yellow[0]==0):
        numero_yellow[0] = numero_yellow[0] + (control_yellow-numero_controlyellow[0])

def numerototalgreen(control_green):
    numero_controlgreen[0] = numero_controlgreen[1]    
    numero_controlgreen[1] = control_green
    if(control_green > numero_controlgreen[0] or numero_green[0]==0):
        numero_green[0] = numero_green[0] + (control_green-numero_controlgreen[0])

def numerototal4x2(control_4x2):
    numero_control4x2[0] = numero_control4x2[1]    
    numero_control4x2[1] = control_4x2
    if(control_4x2 > numero_control4x2[0] or numero_4x2[0]==0):
        numero_4x2[0] = numero_4x2[0] + (control_4x2-numero_control4x2[0])

def numerototal2x2(control_2x2):
    numero_control2x2[0] = numero_control2x2[1]    
    numero_control2x2[1] = control_2x2
    if(control_2x2 > numero_control2x2[0] or numero_2x2[0]==0):
        numero_2x2[0] = numero_2x2[0] + (control_2x2-numero_control2x2[0])

def numerototal4x2red(control_4x2red):
    numero_control4x2red[0] = numero_control4x2red[1]    
    numero_control4x2red[1] = control_4x2red
    if(control_4x2red > numero_control4x2red[0] or numero_4x2red[0]==0):
        numero_4x2red[0] = numero_4x2red[0] + (control_4x2red-numero_control4x2red[0])

def numerototal4x2yellow(control_4x2yellow):
    numero_control4x2yellow[0] = numero_control4x2yellow[1]    
    numero_control4x2yellow[1] = control_4x2yellow
    if(control_4x2yellow > numero_control4x2yellow[0] or numero_4x2yellow[0]==0):
        numero_4x2yellow[0] = numero_4x2yellow[0] + (control_4x2yellow-numero_control4x2yellow[0])

def numerototal4x2green(control_4x2green):
    numero_control4x2green[0] = numero_control4x2green[1]    
    numero_control4x2green[1] = control_4x2green
    if(control_4x2green > numero_control4x2green[0] or numero_4x2green[0]==0):
        numero_4x2green[0] = numero_4x2green[0] + (control_4x2green-numero_control4x2green[0])


def numerototal2x2red(control_2x2red):
    numero_control2x2red[0] = numero_control2x2red[1]    
    numero_control2x2red[1] = control_2x2red
    if(control_2x2red > numero_control2x2red[0] or numero_2x2red[0]==0):
        numero_2x2red[0] = numero_2x2red[0] + (control_2x2red-numero_control2x2red[0])

def numerototal2x2yellow(control_2x2yellow):
    numero_control2x2yellow[0] = numero_control2x2yellow[1]    
    numero_control2x2yellow[1] = control_2x2yellow
    if(control_2x2yellow > numero_control2x2yellow[0] or numero_2x2yellow[0]==0):
        numero_2x2yellow[0] = numero_2x2yellow[0] + (control_2x2yellow-numero_control2x2yellow[0])

def numerototal2x2green(control_2x2green):
    numero_control2x2green[0] = numero_control2x2green[1]    
    numero_control2x2green[1] = control_2x2green
    if(control_2x2green > numero_control2x2green[0] or numero_2x2green[0]==0):
        numero_2x2green[0] = numero_2x2green[0] + (control_2x2green-numero_control2x2green[0])


def countrs(arraymask,image):
    array,name = color_range()
    ar = 1500 

    control = 0
    control_red = 0
    control_yellow = 0
    control_green = 0
    control_4x2 = 0
    control_2x2 = 0
    control_4x2red = 0
    control_4x2yellow = 0
    control_4x2green = 0
    control_2x2red = 0
    control_2x2yellow = 0
    control_2x2green = 0
    
    for k in range(len(name)):
        contours, hierarchy = cv.findContours(arraymasks[k], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if (area > ar):
                #idnumber=idnumber+1
                x, y, w, h = cv.boundingRect(contour)
                cv.drawContours(image, contours, -1, (0, 255, 0), 5)
                image = cv.putText(image, name[k], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)
                image = cv.putText(image, str(int(((h))/22))+'x'+str(int(((w))/22)), (x+100, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)
                #image = cv.putText(image, str('ID:' + int(idnumber)), (x+200, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)
                #print(w)
                #print(h)
                control = control + 1
                if(name[k]=='Red'):
                    control_red = control_red + 1
                if(name[k]=='Yellow'):
                    control_yellow = control_yellow + 1
                if(name[k]=='Green'):
                    control_green = control_green + 1
                if(int(((h))/23)==4 and int(((w))/23)==2):
                    control_4x2 = control_4x2 + 1    
                if(int(((h))/23)==2 and int(((w))/23)==2):
                    control_2x2 = control_2x2 + 1
                if((int(((h))/23)==4 and int(((w))/23)==2) and name[k]=='Green'):
                    control_4x2green = control_4x2green + 1 
                if((int(((h))/23)==4 and int(((w))/23)==2) and name[k]=='Red'):
                    control_4x2red = control_4x2red + 1
                if((int(((h))/23)==4 and int(((w))/23)==2) and name[k]=='Yellow'):
                    control_4x2yellow = control_4x2yellow + 1
                if((int(((h))/23)==2 and int(((w))/23)==2) and name[k]=='Green'):
                    control_2x2green = control_2x2green + 1 
                if((int(((h))/23)==2 and int(((w))/23)==2) and name[k]=='Red'):
                    control_2x2red = control_2x2red + 1
                if((int(((h))/23)==2 and int(((w))/23)==2) and name[k]=='Yellow'):
                    control_2x2yellow = control_2x2yellow + 1
                
    numerototal(control)
    numerototalred(control_red)
    numerototalyellow(control_yellow)
    numerototalgreen(control_green)
    numerototal4x2(control_4x2)
    numerototal2x2(control_2x2)
    numerototal4x2red(control_4x2red)
    numerototal4x2yellow(control_4x2yellow)
    numerototal4x2green(control_4x2green)
    numerototal2x2red(control_2x2red)
    numerototal2x2yellow(control_2x2yellow)
    numerototal2x2green(control_2x2green)
    print("\nInformação atual")
    print("Numero de peças: "+str(int(control)))
    print("Numero de peças Vermelhas: "+str(int(control_red)))
    print("Numero de peças Amarelas: "+str(int(control_yellow)))
    print("Numero de peças Verdes: "+str(int(control_green)))
    print("Numero de peças 4x2: "+str(int(control_4x2)))
    print("Numero de peças 2x2: "+str(int(control_2x2)))
    print("Numero de peças Vermelhas 4x2 : "+str(int(control_4x2red)))
    print("Numero de peças Amarelas 4x2 : "+str(int(control_4x2yellow)))
    print("Numero de peças Verdes 4x2 : "+str(int(control_4x2green)))
    print("Numero de peças Vermelhas 2x2 : "+str(int(control_2x2red)))
    print("Numero de peças Amarelas 2x2 : "+str(int(control_2x2yellow)))
    print("Numero de peças Verdes 2x2 : "+str(int(control_2x2green)))
    print("\nHistorico")
    print("Numero total de peças: "+str(int(numero_red[0]+numero_yellow[0]+numero_green[0])))
    print("Numero total de peças Vermelhas: "+str(int(numero_red[0])))
    print("Numero total de peças Amarelas: "+str(int(numero_yellow[0])))
    print("Numero total de peças Verdes: "+str(int(numero_green[0])))
    print("Numero total de peças 4x2: "+str(int(numero_4x2[0])))
    print("Numero total de peças 2x2: "+str(int(numero_2x2[0])))
    print("Numero total de peças Vermelhas 4x2: "+str(int(numero_4x2red[0])))
    print("Numero total de peças Amarelas 4x2: "+str(int(numero_4x2yellow[0])))
    print("Numero total de peças Verdes 4x2: "+str(int(numero_4x2green[0])))
    print("Numero total de peças Vermelhas 2x2: "+str(int(numero_2x2red[0])))
    print("Numero total de peças Amarelas 2x2: "+str(int(numero_2x2yellow[0])))
    print("Numero total de peças Verdes 2x2: "+str(int(numero_2x2green[0])))
    return  image

#main
while(1):
    camara()
    name = 'teste.jpeg'
    image = cv.imread(name,5)
    imagecountrs = cv.imread(name)

    imagefilter = filterblur(image)
    hsvFrame = cv.cvtColor(imagefilter,cv.COLOR_BGR2HSV)
    arraymasks = masks(hsvFrame)

    image = countrs(arraymasks,imagecountrs)

    
    MostrarImage(image,"original")

    k = cv.waitKey(5)
    if k ==27:
        break
cap.release()
cv.destroyAllWindows()