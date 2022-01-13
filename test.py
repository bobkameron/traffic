import cv2 , numpy

IMG_WIDTH = 30
IMG_HEIGHT = 30

print("what's up buddy \
how's it going man       \
      ")

newlist = [ 3434,'343', 'sadfasf' ,
33434, 444, 'fff ' ]
print (newlist)


newlist = newlist + [3434,343434,'545345345','asdfasdf']

print (newlist) 

new = numpy.array(  [233,3434] )
old = numpy.array(  [3434,5555] )

new2 = new . dot( old) 
print( new2) 

img = cv2.imread ('gtsrb-small/2/00000_00000.ppm', 0 )
img1 = cv2.imread ( 'gtsrb-small/2/00000_00000.ppm', 1) 
img2 = cv2.imread ( 'gtsrb-small/2/00000_00000.ppm', -1 )

#cv2.imshow('imagegray' , img )
#cv2.imshow('imagecolor' , img1)
#cv2.imshow('imageunchanged', img2) 

pixel = img2 [2][2]
pix_gray = img [2][2]
pix_color = img1[2][2]
print(pixel,pix_gray, pix_color ) 

img2resize = cv2.resize ( img2, (2,2) )

img1resize  = cv2.resize ( img2, (2,2) )


cv2.imshow('imageresize',img2resize)

print( img2resize )
print ( img1resize )

imgcombine = img1resize + img2resize
print(imgcombine) 

print( imgcombine [1][1][1])

