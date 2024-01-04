import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

#Create two columns with different width
col1, col2 = st.columns( [0.8, 0.2])
with col1:               # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Arial'; color: #FF6633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)

#Add a header and expander in side bar
st.sidebar.markdown('<p class="font">Grobest digital shrimp size measurement tool</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
     st.write("""
        The first AI-based tool to support our customer!
     """) 
    
#Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
weight = st.number_input('Insert the weight')
day = st.number_input('Insert the day')
option = st.selectbox('Orientation of the image',('H','W'))

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)  # Read as grayscale
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as color image
    #image = cv2.imread(uploaded_file.name)
    
    # Load Aruco detector
    parameters = cv2.aruco.DetectorParameters_create()
    #aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

    # dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    # parameters =  cv2.aruco.DetectorParameters()
    # detector = cv2.aruco.ArucoDetector(dictionary, parameters)

   
    model = YOLO('last.pt')
    results = model(image)[0]
    # st.image(uploaded_file)
    # st.image(st.session_state.img,caption="size measurement")
    #st.write(img.shape)



    # Load Image
    # img = cv2.imread(image)

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    #corners, _, _ = cv2.aruco.ArucoDetector.detectMarkers(img, dictionary, parameters)
    #corners, markerIds, rejectedCandidates = detector.detectMarkers(img)

    # Draw polygon around the marker
    int_corners = np.int0(corners)
    cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

    # Aruco Perimeter
    aruco_perimeter = cv2.arcLength(corners[0], True)

    # Pixel to cm ratio
    pixel_cm_ratio = aruco_perimeter / 20


    #contours = detector.detect_objects(img)
    #boxes = results.boxes

    boxes = results.boxes
    xywh = boxes.xywh
    xyxy = boxes.xyxy
    xywh = xywh.numpy()
    xyxy =xyxy.numpy()



    # Draw objects boundaries
    for i in range(xyxy.shape[0]):
        x1 = xywh[i][0]
        x2 = xywh[i][0]+xywh[i][2]
        y1 = xywh[i][1]
        y2 = xywh[i][1]+xywh[i][3]

        if option== 'W':
            object_width = (xywh[i][2] / pixel_cm_ratio)*0.74
            object_weight = (0.002*(object_width**4) - 0.0578*(object_width**3) + 0.7526*(object_width**2) - (3.5356*object_width) + 5.8716)
            object_count = int(1000/object_weight)
            PDG = (object_weight - weight)/day
        elif option== 'H':
            object_width = (xywh[i][3] / pixel_cm_ratio)*0.74
            object_weight = (0.002*(object_width**4) - 0.0578*(object_width**3) + 0.7526*(object_width**2) - (3.5356*object_width) + 5.8716)
            object_count = int(1000/object_weight)
            PDG = (object_weight - weight)/day
        else:
            st.write('input in the Orientation')

        cv2.putText(image, "{} cm".format(round(object_width, 1)), (int(x1 - 50), int(y1 + 120)), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 6)
        cv2.putText(image, "{} g".format(round(object_weight, 1)), (int(x1 - 50), int(y1 + 175)), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 6)
        cv2.putText(image, "{} count".format(round(object_count, 1)), (int(x1 - 50), int(y1 + 205)), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 6)
        cv2.putText(image, "{} pdg".format(round(PDG, 1)), (int(x1 - 50), int(y1 + 245)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="size measurement")
