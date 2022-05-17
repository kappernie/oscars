# from configparser import Interpolation
# from email.mime import image
# from multiprocessing import connection
# from pickle import NONE
# from turtle import color
import streamlit as st 
import mediapipe as mp 
import cv2 
import numpy as np 
# import tempfile 
# import time 
# import PIL
from PIL import Image 
import random


from tensorflow.keras.models import load_model



class infer():
    def __init__(self):

        self.face_count = 0
        self.max_faces = 1
        self.detection_confidence = 0.8
        self.trac_conf = 0.8
        self.celebnames = ['chris Rock','James Brown','Hanna reyes' ,'bob ,nimble' ,'Zendaya','Zoë Kravitz','Venus Williams',
                        'Rachel Zegler','Timothée Chalamet','Kristen Stewart','Jessica Chastain', 'Ariana DeBose','Billie Eilish',
                        'Cynthia Erivo','Penelope Cruz','Javier Bardem','Kourtney Kardashian','Travis Barker','Simu Liu','Serena Williams','Nicole Kidman',
                        'Laverne Cox','Maddie Ziegler','Lily James','Kirsten Dunst','Lupita Nyong o','David Oyelowo',
                        'Ruth E. Carter','Wesley Snipes','Alana Haim','Este Haim','Danielle Haim','Jada Pinkett Smith',
                        'Will Smith','Amy Schumer','Aunjanue Ellis','Amy Forsyth','Jane Forsyth','Rita Moreno','Olivia Colman','Benedict',
                        'Sophie Hunter','Judi Dench','Ava DuVernay','Marlee Matlin','Niecy Nash']
        

        # Acftor halfmasks
        self.actorfilter0 = cv2.imread('./halfface/maggie.jpeg', cv2.IMREAD_UNCHANGED)
        self.actorfilter1 = cv2.imread('./halfface/garfield.jpg', cv2.IMREAD_UNCHANGED)
        self.actorfilter2 = cv2.imread('./halfface/ashton.jpg', cv2.IMREAD_UNCHANGED)
        self.actorfilter3 = cv2.imread('./halfface/ben.png', cv2.IMREAD_UNCHANGED)
        self.actorfilter4 = cv2.imread('./halfface/jill.jpg', cv2.IMREAD_UNCHANGED)
        self.actorfilter5 = cv2.imread('./halfface/mia.jpg', cv2.IMREAD_UNCHANGED)
        self.actorfilter6 = cv2.imread('./halfface/latifa.jpg', cv2.IMREAD_UNCHANGED)
        self.actorfilter7 = cv2.imread('./halfface/hadish.png', cv2.IMREAD_UNCHANGED)
        self.actorfilter8 = cv2.imread('./halfface/will.jpg', cv2.IMREAD_UNCHANGED)
        self.actorfilter9 = cv2.imread('./halfface/gabri.jpg', cv2.IMREAD_UNCHANGED)


        # actor names 
        self.actorfiltername0 = "Maggie Gyllenhaal"
        self.actorfiltername1 = "Andrew Garfield"
        self.actorfiltername2 = "Ashton Kutcher"
        self.actorfiltername3 = "Ben Cumberbatch"
        self.actorfiltername4 =  "jill scot"
        self.actorfiltername5 = "Mia kunis"
        self.actorfiltername6 = "Queen latifah"
        self.actorfiltername7 = "Tiffany hadish"
        self.actorfiltername8 = "Will Smith"
        self.actorfiltername9 = "Tati gabrielle"


        #actor faces for matching 
        self.actorface0 = cv2.imread('./fullface/maggie.jpeg', cv2.IMREAD_UNCHANGED)
        self.actorface1 = cv2.imread('./fullface/garfield.jpg', cv2.IMREAD_UNCHANGED)
        self.actorface2 = cv2.imread('./fullface/ashton.jpg', cv2.IMREAD_UNCHANGED)
        self.actorface3 = cv2.imread('./fullface/ben.png', cv2.IMREAD_UNCHANGED)
        self.actorface4 = cv2.imread('./fullface/jill.jpg', cv2.IMREAD_UNCHANGED)
        self.actorface5 = cv2.imread('./fullface/mia.jpeg', cv2.IMREAD_UNCHANGED)
        self.actorface6 = cv2.imread('./fullface/latifa.jpg', cv2.IMREAD_UNCHANGED)
        self.actorface7 = cv2.imread('./fullface/hadish.png', cv2.IMREAD_UNCHANGED)
        self.actorface8 = cv2.imread('./fullface/will.jpg', cv2.IMREAD_UNCHANGED)
        self.actorface9 = cv2.imread('./fullface/gabri.jpg', cv2.IMREAD_UNCHANGED)


        self.celebfaces = [self.actorfilter0,self.actorfilter1,self.actorfilter2,self.actorfilter3,self.actorfilter4,
        self.actorfilter5,self.actorfilter6,self.actorfilter7,self.actorfilter8,self.actorfilter9]

        # mediapipe configs
        self.drawing_utils = mp.solutions.drawing_utils
        self.face_mesh = mp.solutions.face_mesh
        self.specifications = self.drawing_utils.DrawingSpec(thickness = 1, circle_radius = 1)

        
        # self.model = load_model('./output/siamese_model') #load trained model

    def applyFilter(self, source, imageFace, dstMat):
                (imgH, imgW) = imageFace.shape[:2]

                #filterImg = cv2.resize(filterImg,(face_width,face_height))
                # grab the spatial dimensions of the source image and define the
                # transform matrix for the *source* image in top-left, top-right,
                # bottom-right, and bottom-left order
                (srcH, srcW) = source.shape[:2]          
                srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

                # compute the homography matrix and then warp the source image to the
                # destination based on the homography
                (H, _) = cv2.findHomography(srcMat, dstMat)
                warped = cv2.warpPerspective(source, H, (imgW, imgH))


                # Split out the transparency mask from the colour info
                overlay_img = warped
                # [:,:,:3] # Grab the BRG planes
                overlay_mask = warped
                # [:,:,3:]  # And the alpha plane
            
                # Again calculate the inverse mask
                # background_mask = 255 - overlay_mask
            
                # Turn the masks into three channel, so we can use them as weights
                overlay_mask = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                # background_mask = cv2.cvtColor(background_mask, cv2.COLOR_BGR2RGB)
            
                # Create a masked out face image, and masked out overlay
                # We convert the images to floating point in range 0.0 - 1.0
                face_part = (imageFace) 
                overlay_part = overlay_mask 
                output = np.uint8(cv2.addWeighted(overlay_part, 1,face_part , 1, 0.0))
                return output
    

    def predict(self,img1,img2,name,halfmask):
        # make copies of images 
        imageA = img1.copy()
        imageB = cv2.imread(img2)  #img2 is already loaded into memory

        # load both the images and convert them to grayscale
        # imageA = img1
        # imageB = img2  #img2 is already loaded into memory

        # add channel a dimension to both the images
        imageA = np.expand_dims(imageA, axis=-1)
        imageB = np.expand_dims(imageB, axis=-1)

        # add a batch dimension to both images
        imageA = np.expand_dims(imageA, axis=0)
        imageB = np.expand_dims(imageB, axis=0)

        # scale the pixel values to the range of [0, 1]
        imageA = imageA / 255.0
        imageB = imageB / 255.0

        # use our siamese model to make predictions on the image pair,
        # indicating whether or not the images belong to the same class
        preds = self.model.predict([imageA, imageB])

        if preds[0][0] >= 0.5 :
            return img1 ,name,halfmask

    def image_inference(self,img ,name = None ,halfmask = None):
       
        with self.face_mesh.FaceMesh(
            # static_image_mode = True,
            max_num_faces = self.max_faces,
            min_detection_confidence = self.detection_confidence,
            min_tracking_confidence = self.trac_conf
        ) as mesh:

            metrics = mesh.process(img)
            output = img.copy()
            image = img

            try: 

                height, width, _ = img.shape
                for faces in metrics.multi_face_landmarks:
                    pt = faces.landmark[10]
                    x = int(pt.x * width)
                    y = int(pt.y * height)

                # get destination matrix 
                topLeft = [float(faces.landmark[338].x * image.shape[1]) - (float(faces.landmark[338].x * image.shape[1])*0.10), float(faces.landmark[338].y * image.shape[0]) - (float(faces.landmark[338].y * image.shape[0]))*0.10]
                topRight = [float(faces.landmark[284].x * image.shape[1]) + (float(faces.landmark[284].x * image.shape[1])*0.10) , float(faces.landmark[284].y * image.shape[0]) - (float(faces.landmark[284].y * image.shape[0])*0.10)]
                bottomRight = [float(faces.landmark[365].x * image.shape[1])+ (float(faces.landmark[365].x * image.shape[1])*0.10) , float(faces.landmark[365].y * image.shape[0]) + (float(faces.landmark[365].y * image.shape[0])*0.10)]
                bottomLeft = [float(faces.landmark[377].x * image.shape[1]) - (float(faces.landmark[377].x * image.shape[1])*0.10) , float(faces.landmark[377].y * image.shape[0]) + (float(faces.landmark[377].y * image.shape[0])*0.10)]
                dstMat = [ topLeft, topRight, bottomRight, bottomLeft ]
                dstMat = np.array(dstMat)

                
                radius = 20
                start_point = ( x- 150,y- 220 )
                end_point = ( x + 170,y - 130 )
                color = (0, 0, 0)
                thickness = -1
              
                # if name is not None :
                #     output = self.applyFilter(halfmask, img , dstMat)
                #     celebname = name
                # else:
                #     # To improve performance, optionally mark the image as not writeable to
                #     # image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                celeb_faces = random.randrange(len(self.celebfaces))
                output = self.applyFilter( self.celebfaces[celeb_faces] ,image , dstMat)
                celeb_name = random.randrange(len(self.celebnames))
                celebname = f'{self.celebnames[celeb_name]}'

                cv2.circle(output, (x, y - 100), 5, (100, 100, 0), -1)
                cv2.rectangle(output, start_point, end_point, color, thickness)
                cv2.putText(output, celebname, (x - 80 , y- 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                return cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
            except Exception as e:
                return None



st.title('Which Oscars Actor do you resemble?')

st.markdown(""" 
<style>
[data-testid = "stSidebar"][aria-expanded = "true"] > div:first-child
{width: 350px}
[data-testid = "stSidebar"][aria-expanded = "false"] > div:first-child
{width: 350px}
</style>
""",unsafe_allow_html = True)



st.sidebar.title('Face recogniton apps')
st.sidebar.subheader('The apps below allow you to check which Oscars actor you look like ')

@st.cache()
def resizeImg(img,width = None ,height = None ,inter = cv2.INTER_AREA):
    dim = None
    (h,w)  = img.shape[:2] 
    if height is None:
         return img
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    r = width/float(w)
    dim = (width,int(h*r))
    resizedimg = cv2.resize(img,dim,interpolation = inter)
    return resizedimg

app = st.sidebar.selectbox('Select app you want to use',['Face','video','About'])

if app == 'Face':
    global pic
    st.markdown('<h3>upload or take a picture to check which actor you resemble</h3>' ,unsafe_allow_html=True)
    b1,b2,b3 = st.columns(3)
    with b1:
        pic1 = st.camera_input('Take a picture')
    with b2 :
        pic = st.file_uploader('upload an image with a face', type = ['jpeg','jpg','png'])
    with b3 :
        # pic2 = cv2.imread('img.jpg')
        st.button('inference')
        if pic :
           for i in range(10):
            originalImg,actorname,actormaskmask = infer().predict(np.array(Image.open(pic)) ,\
                 f'self.actorface{i}', f'self.actorfiltername{i}', f'self.actorfilter{i}')
            result = infer().image_inference(originalImg,actorname,actormaskmask)
           if result is not None:
                st.image(result)
           else:
               st.error('no face found')
        elif pic1 :
            for i in range(10):
                originalImg,actorname,actormaskmask = infer().predict(np.array(Image.open(pic1)) ,\
                    f'self.actorface{i}', f'self.actorfiltername{i}', f'self.actorfilter{i}')

                result = infer().image_inference(originalImg,actorname,actormaskmask)
            if result is not None:
                st.image(result)
            else:
                st.error('no face found')
        else:
            st.error('no image uploaded')

if  app == 'About':
    with st.expander('test1') :

        st.write("""
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
        """)
        
    with st.expander('test2') :
        st.write('yes')
    with st.expander('test3') :
        st.write('no')

if  app == 'video':
   global recordvid
   global recordgif

   c1,c2,c3 = st.columns(3)
   d1,d2,d3 = st.columns(3)
   with c1:
        run = st.checkbox('Show-webcam')
   with c2:
        recordgif = st.checkbox('recordGIF')
   with c3:
        recordvid = st.checkbox('recordVID')



   frames = st.image([])
   cam = 0
   cap = cv2.VideoCapture(cam)

   while  run :
       ret,frame = cap.read()
       frame = cv2.cvtColor(cv2.flip(frame, 1) ,cv2.COLOR_BGR2RGB)
    #    try:
    #     frame = image_inference(frame)
    #     frames.image(frame)
    #    except PIL.UnidentifiedImageError:
    #        pass
       if infer().image_inference(frame) is None :
           frames.image(frame)
       else:
           frames.image(infer().image_inference(frame))

        
       
       
   else: 
        st.error('camera stopped')
        cap.release()
#    with d1:
#     while  recordgif :
#         pass
#     else: 
#             st.download_button('download recorded gif' ,'0.jpg')
#             cap.release()
#    with d2:
#     while  recordvid :
#         pass
#     else: 
#             st.download_button('download recorded video' ,'0.jpg')
#             cap.release()
    

# st.download_button('download recorded gif')
# st.download_button('download recorded video')

