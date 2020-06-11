import pandas as pd
import cv2
import numpy as np
import os
fer_data=pd.read_csv('fer2013.csv',delimiter=',')



for index,row in fer_data.iterrows():
    pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
    img=pixels.reshape((48,48))
    pathname=os.path.join('Output',str(index)+'.jpg')
    cv2.imwrite(pathname,img)
    print('image saved ias {}'.format(pathname))
