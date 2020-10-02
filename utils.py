import os
from os.path import exists

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import time
from datetime import datetime
import pickle
from tqdm import tqdm

from csv import writer

from PIL import Image
import piexif
import cv2

from mtcnn import MTCNN

import ast # str to python expression (e.g. dict)

from keras.models import load_model

from keras.preprocessing.image import img_to_array

from sklearn.manifold import TSNE

from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# Helper Functions----------------------------------------------------------------------------------------------------

def get_jpg_filenames(directory):
    """Function returns jpg (ignoring other files) files paths in the directory and its subdirectories

    Returns:
        file_paths: list 
    """
    jpg_extensions = ('.jpg', '.JPG', '.jpe', '.jpeg', '.jif', '.jfif', '.jfi')
    list_of_jpg = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        list_of_jpg   += [os.path.join(dirpath, file) for file in filenames if file.endswith(jpg_extensions)]
    return list_of_jpg

def get_creation_time(full_path):
    """Read creation datetime from jpg file metadata

    Args:
        full_path (path): path to the file

    Returns:
        creation datetime: string
    """
    image = Image.open(full_path)
    exif = image.getexif()
    creation_date = exif.get(36867)
    return creation_date

def get_creation_times(list_of_jpg):
    """Function returns a list of creation datetimes for the files in the list of jpg files paths

    Args:
        list_of_jpg (list): jpg files paths

    Returns:
        creation datetime list: list of strings
    """    
    list_of_creation_times = []
    for jpg in list_of_jpg:
        try:
            image = Image.open(jpg)
            exif = image.getexif()
            creation_time = exif.get(36867)
        except:
            creation_time = None
        list_of_creation_times.append(creation_time)
    return list_of_creation_times

def change_jpg_datetime(filename, year, month, day):
    """Writes new datetime to metadata of a jpg file

    Args:
        filename (path): path to the file
        year (int): year, 4 digits
        month (int): month, 1-2 digits
        day (int): day, 1-2 digits
    """    
    exif_dict = piexif.load(filename)
    exif_dict['Exif'] = { piexif.ExifIFD.DateTimeOriginal: datetime(year, month, day, 0, 0, 0).strftime("%Y:%m:%d %H:%M:%S") }
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, filename)

def change_jpg_list_datetime(jpg_filenames_list, year, month, day):
    """Change metadata for a list of jpg files to specific date

    Args:
        jpg_filenames_list (list): list of jpg files paths
        year (int): year, 4 digits
        month (int): month, 1-2 digits
        day (int): day, 1-2 digits
    """    
    for filename in jpg_filenames_list:
        exif_dict = piexif.load(filename)
        exif_dict['Exif'] = {piexif.ExifIFD.DateTimeOriginal: datetime(year, month, day, 0, 0, 0).strftime("%Y:%m:%d %H:%M:%S")}
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, filename)

def change_datetime_in_folder(directory, Year, Mo, Da):
    """Change metadata for a jpg files in the directory to specific date

    Args:
        jpg_filenames_list (list): list of jpg files paths
        year (int): year, 4 digits
        month (int): month, 1-2 digits
        day (int): day, 1-2 digits
    """    
    jpg_filenames_list = get_jpg_filenames(directory)
    change_jpg_list_datetime(jpg_filenames_list, Year, Mo, Da)  
        
def creation_times_range(creation_times_list):
    """Obtain range (period of time) from list of files creation times

    Args:
        creation_times_list (list of strings): creation datetime (from jpg files metadata)
    """    
    if creation_times_list != []:
        try:
            years =  [int(x[0:4]) for x in creation_times_list if x != None]
            months =  [int(x[5:7]) for x in creation_times_list if x != None]
            days = [int(x[8:10]) for x in creation_times_list if x != None]
            print('years:', min(years),'-', max(years),
                  '; months:',min(months),'-',max(months),
                  '; days:', min(days),'-',max(days))
        except:
            print('Warning!!! Some files with unknown time')
        if None in creation_times_list:
                print('Warning!!! Some files with creation time == None in the directory')

def get_creation_times_range(directory):
    """Obtain range of creation times for jpg files in the directory

    Args:
        directory (path): path to directory

    Returns:
        creation times range (string): creation times: from - to
    """    
    jpg_filenames_list = get_jpg_filenames(directory)
    creation_times_list = get_creation_times(jpg_filenames_list)
    return creation_times_range(creation_times_list)

def print_creation_times_for_subfolders(directory):
    """Prints creation times range for files in subfolders in a directory

    Args:
        directory (path): path to directory with subfolders holding jpg files
    """    
    list_subfolders_with_paths = [f.path for f in os.scandir(directory) if f.is_dir()]
    print('Number of immediate subfolders: ', len(list_subfolders_with_paths))
    print()
    for idx, subdirectory in enumerate(list_subfolders_with_paths):
        print(idx)
        print(subdirectory)
        jpg_filenames_list = get_jpg_filenames(subdirectory)
        print(len(jpg_filenames_list), 'jpg files in subdirectory')
        creation_times_list = get_creation_times(jpg_filenames_list)
        creation_times_range(creation_times_list)
        print()

# 1. MTCNN functions---------------------------------------------------------------------------------------------------

def read_image(file_path):
    """Read jpg image file using cv2

    Args:
        file_path (path): path to the file

    Returns:
        image: 
    """
    imageBGR = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR) # Use to open unicode folders/files. https://stackoverflow.com/questions/43185605/how-do-i-read-an-image-from-a-path-with-unicode-characters
    image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB) # cv2 uses BGR color ordering, so need to change order for RGB. https://stackoverflow.com/questions/52494592/wrong-colours-with-cv2-imdecode-python-opencv
    return image 

def try_create_csv_file(MTCNN_results_file):
    """Creates new csv file if it does not exist

    Args:
        full_path (path): path to the file

    Returns:
        image: 
    """
    # Create CSV file for MTCNN results
    if exists(MTCNN_results_file):
        pass
    else:
        row = 'jpg_filename,creation_time,MTCNN_result\n'
        with open(MTCNN_results_file,'a') as file:
            file.write(row)
        
def append_list_as_row(file_name, list_of_elem):
    """Appends new row to a csv file

    Args:
        file_name (path): path to the file
        list_of_elements (list): list of elements to append to the file        

    """
    # Open file in append mode
    with open(file_name, 'a+', newline='', encoding="utf-8") as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)              

        
def get_mtcnn_results(directory, MTCNN_results_file, min_face_size = 200):
    """Get mtcnn results and creation_time for all jpg files in the directory and its subdirectories
    The function will skip a file if the result for the file already exists in the results file
    The function will skip corrupted files

    Args:
        directory (path): path to the directory
        MTCNN_results_file (path): path to a csv file to store results
        min_face_size (int): minimum face height that MTCNN is looking for (smaller face sizes increase search time)
        
    Output:
        Creates csv file (if it is not created before) and writes results to it
    """
    detector = MTCNN(min_face_size = min_face_size)
    # heavy operation, performs every time when called, don't put it in the for loop
    
    # Define helper function get single MTCNN result
    def get_MTCNN_result(file_path):
        """Get mtcnn result for 1 jpg file
        
        Args:
            file_path (path): file path

        Returns:
            MTCNN result (list):
        """
        image = read_image(file_path)
        return detector.detect_faces(image)     

    # Try create a csv file for results
    try_create_csv_file(MTCNN_results_file)
    # Read a dataframe from the file
    df = pd.read_csv(MTCNN_results_file)

    jpg_filenames_list = get_jpg_filenames(directory)
    print(len(jpg_filenames_list), 'jpg files in directory')
    scanned_jpg_list = df['jpg_filename'].tolist()
    print(len(scanned_jpg_list), 'jpg files already scanned before')    
    

    time.sleep(0.2) # This pause is required so that tqdm operates correctly
    for filename in tqdm(jpg_filenames_list):
        if filename not in scanned_jpg_list:
            # Try is required to skip corrupted files
            try:
                MTCNN_result = get_MTCNN_result(filename)
                if MTCNN_result != []:
                    creation_time = get_creation_time(filename)
                else:
                    creation_time = None
            except:
                creation_time = None
                MTCNN_result = None
            append_list_as_row(MTCNN_results_file, [filename, creation_time, MTCNN_result])
    time.sleep(0.2)

# 2. Filter results functions--------------------------------------------------------------------------------------------------

def load_MTCNN_scan_results(MTCNN_results_file_path):
    """Load results of MTCNN scan from the csv file

    Args:
        MTCNN_results_file_path (path): path to the file with MTCNN results
        
    Returns:
        Results dataframe (pd.DataFrame): DataFrame with columns: file_path (path to the original photo scanned), creation_date, MTCNN results (dict) 
    """
    df = pd.read_csv(MTCNN_results_file_path)
    print(len(df), 'jpg files scanned with MTCNN')
    df.dropna(inplace=True)
    print(len(df), 'jpg files with detected faces')
    
    # Reformat the dataframe so that each face found with MTCNN algorithm is one row of data
    vals = [[]]
    for _, row in df.iterrows():    
        file_path = row['jpg_filename']
        creation_date_str = row['creation_time'][0:10]
        creation_date = datetime.strptime(creation_date_str, '%Y:%m:%d')
        result = ast.literal_eval(row['MTCNN_result']) # string to python data format
        for res in result:
            vals.append([file_path, creation_date, res])
            columns=['file_path','creation_date', 'MTCNN_result']
    df_res=pd.DataFrame(vals, columns = columns)
    df_res.drop(df_res.index[0], inplace=True) # drop first empty row
    df_res.reset_index(drop=True, inplace=True)
    print( len(df_res), 'total faces detected on all photos',)
    return df_res


def variance_of_laplacian(image):
    """Assessment of bluriness of an image. Bigger value - lower bluriness.
    Idea taken from https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    and modified. I found it emperically that (max-min)**2/var is working better than simple variance
    as an assessment of bluriness. 

    Args:
        image
        
    Returns:
        assessment of bluriness
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    conv2d = cv2.Laplacian(gray_image, cv2.CV_64F)    
    return int((conv2d.max() - conv2d.min())**2/(2*conv2d.var())) # int((conv2d.max() - conv2d.min())**2/2), int(conv2d.var()), 

def mark_landmarks(image, keypoints, landmark_color = (0,255,0)):
    """Mark landmarks of keypoints as circles 

    Args:
        image
        keypoints (dict): keypoins (one of the results of MTCNN)
        landmark color (R,G,B): RGB notation
    
    Returns:
        image with marked landmarks
    """
    #cv2.rectangle(image,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),landmark_color,1)
    cv2.circle(image,(keypoints['left_eye']), 1, landmark_color, 2) # circle diameter, landmark_color, line thickness
    cv2.circle(image,(keypoints['right_eye']), 1, landmark_color, 2)
    cv2.circle(image,(keypoints['nose']), 1, landmark_color, 2)
    cv2.circle(image,(keypoints['mouth_left']), 1, landmark_color, 2)
    cv2.circle(image,(keypoints['mouth_right']), 1, landmark_color, 2)
    return image

def rotate_image(image, angle):
    """Rotate image

    Args:
        image
        angle (deg): angle to rotate an image
    
    Returns:
        rotated image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

def get_rotated_keypoints(keypoints, rot_mat):
    """Rotate coordinates of keypoints

    Args:
        keypoints (dict)
        rot_mat: rotation matrix
    
    Returns:
        keypoints in new coordinates
    """    
    def get_rotated_coordinate(rot_mat, old_coord): #https://cristianpb.github.io/blog/image-rotation-opencv
        expanded_coord = old_coord + (1,) 
        product = np.dot(rot_mat,expanded_coord)
        new_coord = int(product[0]), int(product[1])               
        return new_coord
    
    keypoints_rotated = keypoints.copy()
    for key, value in keypoints_rotated.items():
        keypoints_rotated[key] = get_rotated_coordinate(rot_mat, value)
    return keypoints_rotated

def is_not_grayscale(image):
    """Check if image is grayscale

    Args:
        image
    
    Returns:
        Bool: True if not grayscale
    """ 
    return image[:, :, 0].sum() != image[:, :, 1].sum()

def resize_image(image, new_size):
    """Resize image

    Args:
        image
        new_size (int,int): new size of an image
    
    Returns:
        image: rescaled image
    """ 
    image_resized = cv2.resize(image, new_size)
    return image_resized

def image_filter(df, save_image_folder, preview, confidence_filter, face_height_filter, nose_shift_filter, eye_line_angle_filter, sharpness_filter):
    """Image filter can work in preview, save, save/preview mode
    uses several parameters to filter the images
    
    Args:
        df (pd.DataFrame): dataframe with MTCNN results
        save_image (Bool): save filtered image to specific folder
    
    Returns:
        Bool: True if not grayscale
    """
    if save_image_folder == False:
        show_landmarks = True
    else:
        show_landmarks = False    
    
    img_cnt = 0
    good_img_cnt = 0
    df['face_file_path'] = np.NaN

    # iterate over the rows of the dataframe with MTCNN results
    for index, row in tqdm(df.iterrows()):
        try:
            img_cnt += 1
            creation_date = row['creation_date']    
            res = row['MTCNN_result']
            confidence = res['confidence']
            bounding_box = res['box']
            keypoints = res['keypoints']
            upper_left_x, upper_left_y, width, height  = bounding_box

            # change box to square
            side = max(height, width)
            upper_left_x = int(upper_left_x + width/2 - side/2)
            width=height=side

            if (confidence >= confidence_filter) and (height >= face_height_filter):

                # find an angle of line of eyes.
                dY = keypoints['right_eye'][1] - keypoints['left_eye'][1]
                dX = keypoints['right_eye'][0] - keypoints['left_eye'][0]
                angle = np.degrees(np.arctan2(dY,dX))

                # calculate rotation matrix for this anlge around nose as a central point 
                rot_mat = cv2.getRotationMatrix2D(keypoints['nose'], angle, 1.0)

                # calculate new coordinates of keypoints
                keypoints_rotated = get_rotated_keypoints(keypoints, rot_mat)

                # calculate nose shift
                nose_shift = 100*abs((keypoints_rotated['nose'][0] - keypoints_rotated['left_eye'][0] - dX/2)/dX)

                if (nose_shift <= nose_shift_filter) and (abs(angle) <= eye_line_angle_filter):
                    image = read_image(row.file_path)

                    if is_not_grayscale(image):

                        image_rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

                        image_cropped_central_part = image_rotated[int(upper_left_y+height*1/2):int(upper_left_y + height*2/3), int(upper_left_x+width*1/3):int(upper_left_x + width*2/3)]

                        sharpness = variance_of_laplacian(image_cropped_central_part)

                        if sharpness >= sharpness_filter:

                            if show_landmarks == True:

                                image_rotated = mark_landmarks(image_rotated, keypoints_rotated)

                            image_cropped = image_rotated[upper_left_y:upper_left_y + height, upper_left_x:upper_left_x + width]

                            if  preview == True:
                                print(good_img_cnt, 'image from total of', img_cnt, 'scanned images')
                                print(row.file_path)
                                print('angle =',int(angle), '; nose shift =', int(nose_shift), '; sharpness =', int(sharpness))
                                #plt.imshow(image_cropped_central_part)
                                #plt.show()
                                plt.imshow(image_cropped)
                                plt.show()
                            

                            if save_image_folder != False:
                                imagefile_path = save_image_folder +'\\'+ str(good_img_cnt) +'.jpg'
                                cv2.imwrite(imagefile_path, cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR))
                                df['face_file_path'].iloc[good_img_cnt] = imagefile_path
                            
                            good_img_cnt+=1
        except:
            pass
    
    
    return df.dropna()

# 3. FaceNet Embeddings Functions--------------------------------------------------------------------------------------------------

def standardize_image(image):
    """Returns standardized image: (x-mean(X))/std(X)
    
    Args:
        image:
    
    Returns:
        image (standardized)
    """   
    mean, std = image.mean(), image.std()
    return (image - mean) / std

def get_facenet_embedding(image_path, model):
    """Generate FaceNet embeddings
    
    Args:
        image_path (path)
    
    Returns:
        embedding (128 dimension vector)
    """
    image = read_image(image_path)
    image = cv2.resize(image, (160,160))
    image=standardize_image(image)
    input_arr = img_to_array(image) # Convert single image to a batch.
    input_arr = np.array([input_arr])
    return model.predict(input_arr)[0,:]

print('Libraries and functions loaded')
