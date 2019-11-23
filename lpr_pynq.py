
# coding: utf-8

# In[1]:


import sys
import os
from hyperlpr_py3 import pipline as pp
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from hashlib import md5
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
base = BaseOverlay("base.bit")


# In[2]:

Mode = VideoMode(640,480,24)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_BGR)
hdmi_out.start()


# In[3]:

frame_out_w = 1920
frame_out_h = 1080
frame_in_w = 640
frame_in_h = 480


# In[4]:


Sheng = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

plateSheng = {"京":"JING","津":"JINA","沪":"HU","渝":"YUA","蒙":"MENG","新":"XIN","藏":"ZANG","宁":"NING",
                 "桂":"GUIA","黑":"HEI","吉":"JIB","辽":"LIAO","晋":"JINB","冀":"JIA","青":"QING","鲁":"LU",
                 "豫":"YUB","苏":"SU","皖":"WAN","浙":"ZHE","闽":"MIN","赣":"GANA","湘":"XIANG","鄂":"E",
                 "粤":"YUE","琼":"QIONG","甘":"GANB","陕":"SHAN","贵":"GUIB","云":"YUN","川":"CHUAN"}
plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]
fontC = ImageFont.truetype("Font/platech.ttf", 20, 0)  


# In[5]:


def drawPred(frame, label, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((left + 1, top - 38), label, (0, 0, 255), font=fontC)    
    imagex = np.array(img)
    return imagex


def isValidPlate(plate,confidence):
    if confidence > 0.8 and (len(plate) == 7 or len(plate) == 8) and plate[0]  in Sheng:
        return True
    return False

def SimpleRecognizePlate(image):
    images = pp.detect.detectPlateRough(
        image, image.shape[0], top_bottom_padding_rate=0.02)

    res_set = []

    for j, plate in enumerate(images):
        plate, rect, origin_plate = plate
        plate = cv2.resize(plate, (136, 36 * 2))
        plate_type = pp.td.SimplePredict(plate)
        plate_color = plateTypeName[plate_type]

        if (plate_type > 0) and (plate_type < 5):
            plate = cv2.bitwise_not(plate)

        image_rgb = pp.fm.findContoursAndDrawBoundingBox(plate)
        image_rgb = pp.fv.finemappingVertical(image_rgb)
        e2e_plate, e2e_confidence = pp.e2e.recognizeOne(image_rgb)
        if isValidPlate(e2e_plate, e2e_confidence): 
            image = drawPred(image, e2e_plate, int(rect[0]),int(rect[1]),int(rect[0]+rect[2]),int(rect[1]+rect[3]))
            res_set.append([e2e_plate,  
                            plate_color,  
                            e2e_confidence,  
                            (rect[0], rect[1])])  
    return image, res_set


# In[6]:


test_dir = "./test-imgs"  
fw = open("./test-results/No14007mresults.txt", 'w+') 


# In[7]:


videoIn = cv2.VideoCapture(0)


# In[10]:


while True:
    videoIn.release()
    videoIn = cv2.VideoCapture(0)
    videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
    videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
    print("Capture device is open: " + str(videoIn.isOpened()))
    ret, frame_vga = videoIn.read()
    filepath = 'test-imgs/{}.{}'.format(md5().hexdigest(), 'jpg')
    cv2.imwrite(filepath, frame_vga)
    image_result =cv2.imread('test-results/d41d8cd98f00b204e9800998ecf8427e.jpg')
    h = 480   # 指定缩放高度
    w = 640
    image_result = cv2.resize(image_result, (w, h))   
    if (ret):      
        outframe = hdmi_out.newframe()
        outframe[0:480,0:640,:] = image_result[0:480,0:640,:]
        hdmi_out.writeframe(outframe)
    else:
        raise RuntimeError("Failed to read from camera.")
    for f in os.listdir(test_dir):
        try:
            path = os.path.join(test_dir, f)  
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1) 
            h = 1024   # 指定缩放高度
            scale = image.shape[1] / float(image.shape[0])  
            w = int(scale * h)
            image = cv2.resize(image, (w, h))   

            t0 = time.time()
            framedrawed, res = SimpleRecognizePlate(image)  
            tlabel = '%.0f ms' % ((time.time() - t0) * 1000)
            info = f + "\n"
            for r in res:         
                info = info + r[0] + "\n" 

            fw.write(info) 
            cv2.imwrite("./test-results/" + f, framedrawed.astype(np.uint8))  
            print(info[:-1])
            print(tlabel) 
        except Exception as e:
            print(e)      
            continue       


# In[ ]:


fw.close()
cv2.destroyAllWindows()

