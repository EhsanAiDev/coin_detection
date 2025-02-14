import cv2 
from tkinter import *
from ultralytics import YOLO
import supervision as sv


camera = cv2.VideoCapture(0)
price = 0
model = YOLO("model/coin_detection.pt" , verbose=False)
box_anntator = sv.BoundingBoxAnnotator()
label_anntatotr = sv.LabelAnnotator()


def NotFindWindow():
    windows = Tk()
    windows.maxsize(250,100)
    windows.minsize(250,100)
    frame = Frame(windows)
    t1 = Label(frame , text="the camera is not find !" , font=(10,10,"bold"))
    b1 = Button(frame , text="QUIT" , command= lambda : windows.destroy())
    frame.pack(pady=20)
    t1.grid(column=0 , row=0)
    b1.grid(column=0, row=1 , pady=5)
    windows.mainloop()

def preadict(img , model):
    global price 

    img = cv2.resize(img , (320,320))
    detect = model.predict(img , imgsz=320 , conf=0.65)[0]
    detection = sv.Detections.from_ultralytics(detect)

    img = box_anntator.annotate(img , detections=detection)
    img = label_anntatotr.annotate(img , detections=detection , labels= [f"{label} {conf:.2}"for label , conf in zip(detection.data["class_name"] , detection.confidence)])


    img = cv2.resize(img , (480,640))

    if detection :
        coins = [int(i) for i in detection.data["class_name"]]
        price = sum(coins)        
    else :
        price = 0

    return price , img


while camera.isOpened():
    _ , frame= camera.read()
    
    if _ :
        y , x = frame.shape[:2]
        price , frame = preadict(img=frame , model=model)

        cv2.putText(frame, f"price : {price} Toman" , (10,30) ,cv2.FONT_HERSHEY_TRIPLEX , 0.8 , (0,255,0) , 1 )
        cv2.putText(frame, "press the Q to quit" , (int(x/2)-210 , y+130) ,cv2.FONT_HERSHEY_COMPLEX , 0.8 , (0,0,255) , 1 )

        cv2.imshow("frame" , frame)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
    else :
        NotFindWindow()
        break

