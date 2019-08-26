import os
from os import walk
from PIL import Image
from shutil import copyfile

label_idx = 0  #comment after first path search ot folder change
cls_id = '0'
def conv(bound):
    x = (bound[0] + bound[1])/2.0
    y = (bound[2] + bound[3])/2.0
    w = abs(bound[1] - bound[0])
    h = abs(bound[3] - bound[2])
    return (x,y,w,h)

mypath = "./training/Preprocessing/cropped/" #change Accordingly
img_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    for f in filenames:
        base = os.path.basename(f)
        base = os.path.splitext(base)[0]
        img_name_list.append(base)
    break

for i in img_name_list:
    box=[1,2,3,4]
    box[0] = data['points'][0][str(label_idx)][0]['x']
    box[1] = data['points'][0][str(label_idx)][1]['x']
    box[2] = data['points'][0][str(label_idx)][0]['y']
    box[3] = data['points'][0][str(label_idx)][1]['y']
    
    q,w,e,r = conv(box)
    f= open(mypath+i+".txt","w+")
    f.write(cls_id+" "+str(q)+" "+str(w)+" "+str(e)+" "+str(r))
    f.close()
    label_idx+=1
print(label_idx)