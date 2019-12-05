#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util
import shutil
import datetime
import os

def targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_object):

        # make output dir
        i = 0
        output_base_dir = 'C:\\Users\\Aviad\\Desktop\\shrimp\\tensorflow_object_counting_api\\results'
        now = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
        output_dir = os.path.join(output_base_dir, now)
        os.mkdir(output_dir)

        # input video
        cap = cv2.VideoCapture(input_video)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        output_movie = cv2.VideoWriter('the_output_04122019_2.mp4', fourcc, fps, (width, height))

        max_result = 0
        the_result = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                i = i + 1
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      targeted_objects=targeted_object,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=2)
                if(len(the_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, the_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                #cv2.imshow('object counting',input_frame)

                objects_num = int(the_result.split(':')[2])
                

                if objects_num > max_result:
                    shutil.rmtree(output_dir)
                    os.mkdir(output_dir)
                    filename = str(i) + '.png'
                    cv2.imwrite(os.path.join(output_dir,filename),input_frame)
                    max_result = objects_num
                elif objects_num == max_result:
                    filename = str(i) + '.png'
                    cv2.imwrite(os.path.join(output_dir,filename),input_frame)
                    
                output_movie.write(input_frame)
                print ("writing frame")
                print(the_result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            print(max_result)
            cap.release()
            cv2.destroyAllWindows()

def single_image_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled):     
        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')            

       

            input_frame = cv2.imread(input_video)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(input_frame, axis=0)

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # insert information text to video frame
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Visualization of the results of a detection.        
            counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1,input_frame,
                                                                                                1,
                                                                                                is_color_recognition_enabled,
                                                                                                np.squeeze(boxes),
                                                                                                np.squeeze(classes).astype(np.int32),
                                                                                                np.squeeze(scores),
                                                                                                category_index,
                                                                                                use_normalized_coordinates=True,
                                                                                                line_thickness=4)
            if(len(counting_mode) == 0):
                cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
            else:
                cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
            
            # cv2.imshow('tensorflow_object_counting_api',input_frame)
            cv2.imwrite('C:\\Users\\IEUser\\Desktop\\front\\frontend\\temp\\output.png',input_frame)        
            # cv2.waitKey(0)

            return counting_mode
