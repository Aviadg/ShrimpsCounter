#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf
import os

# Object detection imports
from utils import backbone
from api import object_counting_api_short

detection_graph, category_index = backbone.set_model('shrimp_01122019_aug', 'shrimp.pbtxt')

def detect(my_pic_path):

    is_color_recognition_enabled = 0
    result = object_counting_api_short.single_image_object_counting(my_pic_path, detection_graph, category_index, is_color_recognition_enabled) 

    print (result)
    return result
