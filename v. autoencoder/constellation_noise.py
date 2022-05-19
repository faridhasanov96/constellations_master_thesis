#for creating noised image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw
import PIL.Image
import cv2

from IPython import display
import numpy as np
import sys
def draw_circle_white(draw,c,dist):
    r = dist
    shape = [(c[0]-r,c[1]-r),(c[0]+r,c[1]+r)]
    draw.ellipse(shape,fill=250) 
def draw_circle_black(draw,c,dist):
    r = dist
    shape = [(c[0]-r,c[1]-r),(c[0]+r,c[1]+r)]
    draw.ellipse(shape,fill=0) 
    
def generate_image_dotted(edges,dist=40,dot= 2):
    im = PIL.Image.fromarray(edges)
    # create rectangle image 
    draw = ImageDraw.Draw(im)   
    img_shape = edges.shape
    px = im.load()
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if(px[j,i]==255):
                draw_circle_black(draw,(j,i),dist)
                px[j,i]=255
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if(px[j,i]==255):
                draw_circle_white(draw,(j,i),dot)
    return im
def add_noise(im,prob = 0.0001,dot =2 ):
    draw = ImageDraw.Draw(im)
    img_shape = np.array(im).shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if(np.random.random()<prob):
                draw_circle_white(draw,(j,i),dot)
    return np.array(im)

def constellation_create(image_path,dist_btw_dots,radius_of_dots,p_noise):
  image = PIL.Image.open(image_path)
  width,height = image.size
  image = np.array(image.resize((128,128)))
  image = cv2.blur(image, (3,3))
  edges = cv2.Canny(image,80,200)
  dotted = generate_image_dotted(edges,dist_btw_dots,radius_of_dots)
  final = add_noise(dotted,p_noise,radius_of_dots)
  plt.imshow(final)
  return final
from matplotlib import pyplot as plt
%matplotlib inline
import cv2
import numpy as np
def stimuli_dots(image):
    gray = image
    gray = cv2.resize(gray,(800,800))
    #plt.imshow(gray)
    th, threshed = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return cnts
def drawing_figure(image):
    gray = image 
    gray = cv2.resize(gray,(800,800))
    th, threshed = cv2.threshold(gray, 30, 255,cv2.THRESH_BINARY)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    sc =[]
    for ind,c in enumerate(cnts):
        area = cv2.contourArea(c)
        if(area<100): continue
        sc.append(c.copy())
    draw_img = np.zeros(gray.shape)
    for s in sc:
        for pt in s:
            draw_img[pt[0][1],pt[0][0]] =255
    return draw_img
def points_on_image(d_img,dots,lev):
    count = 0
    sel_dots = []
    for dot in dots:
        (x,y),radius = cv2.minEnclosingCircle(dot)
        x1= int(max(x-radius-lev,0))
        x2 = int(min(x+radius+lev,d_img.shape[1]))
        y1= int(max(y-radius-lev,0))
        y2 = int(min(y+radius+lev,d_img.shape[0]))    
        if(np.max(d_img[y1:y2,x1:x2])>10):
            sel_dots.append(dot)
    return sel_dots
def constellation_create(image_path,dist_btw_dots,radius_of_dots,p_noise):
  image = PIL.Image.open(image_path)
  width,height = image.size
  image = np.array(image.resize((256,128)))
  image = cv2.blur(image, (3,3))
  edges = cv2.Canny(image,80,200)
  dotted = generate_image_dotted(edges,dist_btw_dots,radius_of_dots)
  final = add_noise(dotted,p_noise,radius_of_dots)
  #plt.imshow(final, cmap = 'gray')
  return final
