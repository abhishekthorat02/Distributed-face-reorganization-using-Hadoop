

#Python 2.7.2
#Opencv 2.4.2

import cv #Opencv
import Image #Image from PIL
import glob
import os
import cv2
import sys


#frame_count = 0
if len(sys.argv) != 2:
	print "usage: crop_img <base_path>"
        sys.exit(1)
BASE_PATH=sys.argv[1]



def DetectFace(image, faceCascade, returnImage=False):
    # This function takes a grey scale cv image and finds
    # the patterns defined in the haarcascade function
    #variables    
    min_size = (20,20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0

    # Equalize the histogram
    cv.EqualizeHist(image, image)

    # Detect the faces
    faces = cv.HaarDetectObjects(
            image, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )

    # If faces are found
    if faces and returnImage:
        for ((x, y, w, h), n) in faces:
            # Convert bounding box to two CvPoints
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

    if returnImage:
        return image
    else:
        return faces

def pil2cvGrey(pil_im):
    # Convert a PIL image to a greyscale cv image
    pil_im = pil_im.convert('L')
    cv_im = cv.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pil_im.tostring(), pil_im.size[0]  )
    return cv_im

def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)

def faceCrop(imagePattern,boxScale=1):

    faceCascade = cv.Load('haarcascade_frontalface_alt.xml')
    
    global frame_count
    global BASE_PATH
    
    imgList=glob.glob(imagePattern)
    if len(imgList)<=0:
        print 'No Images Found'
        return
    
    for img in imgList:
	
        pil_im=Image.open(img)
	frame_count = img.split("/")[1].split("-")[1].split(".")[0]	
        cv_im=pil2cvGrey(pil_im)
        faces=DetectFace(cv_im,faceCascade)
        if faces:
	    os.mkdir('Cropped'+str(BASE_PATH)+'/'+'frame'+str(frame_count))
            n=1
            for face in faces:
                croppedImage=imgCrop(pil_im, face[0],boxScale=boxScale)
                fname,ext=os.path.splitext(img)
                croppedImage.save('Cropped'+str(BASE_PATH)+'/frame'+str(frame_count)+'/'+str(n)+ext)
		#img1 = cv2.imread('Smile_3_crop1.jpg')
		#res = cv2.resize(img1,None,(100,100))
                n+=1
        else:
            print 'No faces found:', img
    	#frame_count+=1 

def test(imageFilePath):
    pil_im=Image.open(imageFilePath)
    cv_im=pil2cvGrey(pil_im)
    faceCascade = cv.Load('haarcascade_frontalface_default.xml')
    face_im=DetectFace(cv_im,faceCascade, returnImage=True)
    img=cv2pil(face_im)
    img.show()
    img.save('test.png')
    global BASE_PATH
faceCrop('testPics'+str(BASE_PATH)+'/*.jpg',boxScale=1)
