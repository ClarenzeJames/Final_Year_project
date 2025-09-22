import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

import random

import get_images
# import patch_model
import hybrid_model
import whole_model
import patch_model
import effnet_model

getImages = get_images.GetImages()
PatchModel = patch_model.Patch_model()
WholeModel = whole_model.Whole_model()
# p_model = patch_model.PatchModel()
h_model = hybrid_model.HyrbidModel()

def get_data():
    # ----------------- Loading CSV into dataframes -----------------
    print("----------------- Loading CSV into dataframes -----------------")
    raw_train, raw_test = getImages.load_and_combine()

    # ----------------- Getting image file paths -----------------
    print("----------------- Getting image file paths -----------------")
    cropped_train, cropped_test = getImages.load_paths(raw_train, raw_test, image_type='cropped images')
    full_train, full_test = getImages.load_paths(raw_train, raw_test, image_type='full mammogram images')

    # ----------------- Splitting to train, val, test -----------------
    print("----------------- Splitting to train, val, test -----------------")
    new_cropped_train, new_cropped_val, new_cropped_test = getImages.data_split(train_df=cropped_train, test_df=cropped_test)
    new_full_train, new_full_val, new_full_test = getImages.data_split(train_df=full_train, test_df=full_test)

    # ----------------- Creating image and mask pairs -----------------
    print("----------------- Creating image and mask pairs -----------------")
    cropped_pairs = getImages.create_mask_pairs_dataframe(image_type='cropped images')
    full_pairs = getImages.create_mask_pairs_dataframe(image_type='full mammogram images')

    # ----------------- Merge train_df and pairs_df -----------------
    print("----------------- Merge train_df and pairs_df -----------------")
    final_cropped_train, final_cropped_val = getImages.merge_df(train_df=new_cropped_train, 
                                                                val_df=new_cropped_val, 
                                                                pairs_df=cropped_pairs)
    
    final_full_train, final_full_val = getImages.merge_df(train_df=new_full_train, 
                                                          val_df=new_full_val, 
                                                          pairs_df=full_pairs)
    
    # print(final_cropped_train.head(5))
    # print(final_full_train.head(5))

    

    
    #! ONLY RUN ONCE
    save_images(new_cropped_test, new_full_test, final_cropped_train, final_cropped_val, final_full_train, final_full_val)


def save_images(new_cropped_test, new_full_test, final_cropped_train, final_cropped_val, final_full_train, final_full_val):
    print("----------------- Processing and Saving Images -----------------")
    # Dropping NaN values
    new_cropped_test = new_cropped_test.dropna()
    new_full_test = new_full_test.dropna()

    # Dropping duplicate case_ids, keeping the first occurrence
    final_cropped_train = final_cropped_train.drop_duplicates(subset=['case_id'],keep='first')
    final_cropped_val = final_cropped_val.drop_duplicates(subset=['case_id'],keep='first')

    final_full_train = final_full_train.drop_duplicates(subset=['case_id'],keep='first')
    final_full_val = final_full_val.drop_duplicates(subset=['case_id'],keep='first')

    print("----------------- Processing & saving CROPPED IMAGES -----------------")
    # Processing and saving cropped images
    getImages.process_and_save_images(train_set=final_cropped_train, 
                                      val_set=final_cropped_val,
                                      test_set=new_cropped_test, 
                                      image_type='cropped images')
    
    print("----------------- Processing & saving FULL MAMMOGRAM IMAGES -----------------")
    # Processing and saving cropped images
    # getImages.process_and_save_images(train_set=final_full_train, 
    #                                   val_set=final_full_val,
    #                                   test_set=new_full_test, 
    #                                   image_type='full mammogram images')


def train_patch():
    hist1, hist3 = PatchModel.compile_patch_model()

def train_whole_model():
    print("----------------- Trainig Whole model -----------------")
    hist = WholeModel.compile_model()
    # WholeModel.evaluate_on_test()
    # WholeModel.compile_model_kfold()
    # WholeModel.test_set_eval()

# get_data()
# train_patch()
train_whole_model()