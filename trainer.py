import cv2
import numpy as np
import os
from PIL import Image
recog=cv2.face.LBPHFaceRecognizer_create()
path='dataset/faces'
dirname = os.path.dirname(__file__)
myfile= 'dataset/faces/.DS_Store'
if os.path.isfile(myfile):
    os.remove(myfile)

def getimagewithid(path):
    imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagepath in imagepaths:
        faceimage=Image.open(imagepath).convert('L')
        facenp=np.array(faceimage,'uint8')
        print(imagepath)
        id=int(imagepath.split("/")[2].split(".")[1])
        faces.append(facenp)
        ids.append(id)
    return ids,faces
ids,faces=getimagewithid(path)
recog.train(faces,np.array(ids))
recog.save('recog/traingdata.yml')
cv2.destroyAllWindows()
