import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

from sklearn.model_selection import train_test_split

class GetImages():
    def __init__(self):
        self.data_dir = "../../archive"

    def load_and_combine(self):
        # ----------------------- 1. Laoding CSV into dataframes -----------------------

        # Calcification cases
        calc_case_test = pd.read_csv(self.data_dir + '/csv/calc_case_description_test_set.csv')
        calc_case_train = pd.read_csv(self.data_dir + '/csv/calc_case_description_train_set.csv')

        # Mass cases
        mass_case_test = pd.read_csv(self.data_dir + '/csv/mass_case_description_test_set.csv')
        mass_case_train = pd.read_csv(self.data_dir + '/csv/mass_case_description_train_set.csv')

        # Adding new column of abnormality type
        calc_case_test['abnormality_type'] = 'calcification'
        calc_case_train['abnormality_type'] = 'calcification'
        mass_case_test['abnormality_type'] = 'mass'
        mass_case_train['abnormality_type'] = 'mass'

        # Combining the test cases and train cases
        train_df = pd.concat([calc_case_train, mass_case_train], ignore_index=True)
        test_df = pd.concat([calc_case_test, mass_case_test], ignore_index=True)

        # Labeling the data
        def label_data(pathology):
            if pd.isna(pathology):
                return 0
            if 'MALIGNANT' in str(pathology).upper():
                return 1
            else:
                return 0

        train_df['label'] = train_df['pathology'].apply(label_data)
        test_df['label'] = test_df['pathology'].apply(label_data)

        return train_df, test_df
    
    # ----------------------- 2.1 Getting Image file paths -----------------------
    def load_paths(self, train_df, test_df, image_type):
        dicom_data = pd.read_csv(self.data_dir + '/csv/dicom_info.csv')

        # Getting the image type
        dicom_data = dicom_data[dicom_data['SeriesDescription'] == image_type]

        # Matching patient ID to image path and putting it into the train_df/test_df
        patient_to_path = {}
        for _, row in dicom_data.iterrows():
            patient_id = row['PatientID']
            image_path = row['image_path']

            matched = re.search(r'P_\d+', str(patient_id))
            if matched:
                base_id = matched.group()
                if base_id not in patient_to_path:
                    patient_to_path[base_id] = image_path[9:]

        train_df['image_path'] = train_df['patient_id'].map(patient_to_path)
        test_df['image_path'] = test_df['patient_id'].map(patient_to_path)

        print(train_df.columns)

        # Keeping the columns that are wanted
        train_df = train_df[['patient_id','label','image_path','abnormality_type']]
        test_df = test_df[['patient_id','label','image_path','abnormality_type']]

        return train_df, test_df
    
    # ----------------------- 3. Data Splitting -----------------------
    def data_split(self, train_df, test_df):
        # combine all the data first
        combined_df = pd.concat([train_df, test_df], ignore_index=True)

        # Group by patient to ensure all images from same patient stay together
        patient_groups = combined_df.groupby('patient_id')

        patient_ids = list(patient_groups.groups.keys())
        train_patients, temp_patients = train_test_split(
            patient_ids, test_size=0.25, random_state=42,
            stratify=[patient_groups.get_group(p)['label'].iloc[0] for p in patient_ids]
        )

        val_patients, test_patients = train_test_split(
            temp_patients, test_size=0.43, random_state=42
        )

        new_train_df = combined_df[combined_df['patient_id'].isin(train_patients)].copy()
        new_val_df = combined_df[combined_df['patient_id'].isin(val_patients)].copy()
        new_test_df = combined_df[combined_df['patient_id'].isin(test_patients)].copy()

        return new_train_df, new_val_df, new_test_df
    
    # ----------------------- 4. Create image & mask pairs from dicom -----------------------
    def create_mask_pairs_dataframe(self, image_type):
        df = pd.read_csv(self.data_dir + '/csv/dicom_info.csv')
        df['parse_id'] = df['PatientID'].apply(self.parse_patient_id) # parse_patient_id is a helper function
        df = df.dropna(subset=['parse_id'])

        # Create case ID
        df['case_id'] = df.apply(lambda row: 
            f"P_{row['parse_id']['patient_num']}_{row['parse_id']['laterality']}_"
            f"{row['parse_id']['view']}_{row['parse_id']['abnormality_id']}", axis=1)

        pairs = []

        for case_id, group in df.groupby('case_id'):
            base_imgs = group[group['SeriesDescription'] == image_type]
            mask_imgs = group[group['SeriesDescription'] == 'ROI mask images']

            if mask_imgs.empty:
                continue

            mask_path = mask_imgs.iloc[0]['image_path']
            patient_info = mask_imgs.iloc[0]['parse_id']

            # Full mammogram and mask pair
            if not base_imgs.empty:
                pairs.append({
                    'case_id': case_id,
                    'base_image_path': self.data_dir + base_imgs.iloc[0]['image_path'][9:],
                    'mask_image_path': self.data_dir + mask_path[9:],
                    'base_image_type': image_type,
                    'laterality': base_imgs.iloc[0]['Laterality'],
                    'view': base_imgs.iloc[0]['PatientOrientation'],
                    # 'abnormality_type': patient_info.get('abnormality_type'),
                    'dataset': patient_info.get('dataset'),
                    'patient_num': 'P_' + patient_info.get('patient_num')
                })
        return pd.DataFrame(pairs)
    
    # ----------------------- 5. Merging training and pairs dataframe -----------------------
    def merge_df(self, train_df, val_df, pairs_df):

        # Merging the train_df and pairs_df on patient_id
        train_merged_df = pd.merge(train_df, pairs_df, left_on='patient_id', right_on='patient_num', how='left')

        # Merging the val_df and pairs_df on patient_id
        val_merged_df = pd.merge(val_df, pairs_df, left_on='patient_id', right_on='patient_num', how='left')

        # Keeping only the necessary columns
        train_merged_df = train_merged_df[['patient_id', 'label', 'abnormality_type', 
                                'case_id', 'base_image_path', 'laterality', 'view']]
        
        val_merged_df = val_merged_df[['patient_id', 'label', 'abnormality_type', 
                                'case_id', 'base_image_path', 'laterality', 'view']]
        
        train_merged_df = train_merged_df.dropna()
        val_merged_df = val_merged_df.dropna()

        return train_merged_df, val_merged_df
    

    

    
    def process_and_save_images(self, train_set, val_set, test_set, image_type):
        def remove_lines(img):
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            lower = np.array([0,0,200], dtype=np.uint8)
            upper = np.array([255,255,255], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower, upper)

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

            h, w = img.shape
            L = max(15, min(h, w)//30)
            hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
            ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
            hor = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hor_k)
            ver = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ver_k)
            lines = cv2.bitwise_or(hor, ver)

            # Keep only border band (don’t touch in-breast linear anatomy)
            rim = np.zeros_like(lines)
            band = max(6, int(0.03 * min(h, w)))  # 3% of the smaller dimension
            rim[:band, :] = 255; rim[-band:, :] = 255; rim[:, :band] = 255; rim[:, -band:] = 255
            lines = cv2.bitwise_and(lines, rim)

            return lines
        
        def apply_gamma(img):
            gamma = 2.0
            if img.dtype != np.float32 and img.dtype!= np.float64:
                x = img.astype(np.float32) / 255.0
            else:
                x = np.clip(img, 0.0, 1.0).astype(np.float32)
            
            y = np.power(x, gamma)
            y = np.clip(y * 255.0,0,255.0).astype(np.uint8)

            return y
        
        def apply_clahe(img):
            clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(8,8))

            out = clahe.apply(img)
            out = clahe.apply(out)
            return out

        def process_cropped_img(img_path):
            print("process_cropped_img called")
            base_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if base_img is None is None:
                print("Image not found")
                return None
            
            img = apply_gamma(base_img)
            img = apply_clahe(img)

            return img
        
        def process_img(img_path, augment=False):

            print("process_img called")
            # ----------------- Getting Image -----------------
            base_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            plt.imshow(base_img, cmap='gray')
            plt.show()

            if base_img is None is None:
                print("Image not found")
                return None
            
            # ----------------- Removing artifacts -----------------
            if len(base_img.shape) == 3:
                gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = base_img.copy()
            
            # Removal of lines
            lines = remove_lines(gray)
            inv_filtered_img = cv2.bitwise_not(lines)
            inv_filtered_img = cv2.dilate(inv_filtered_img, (5,5))
            img = cv2.bitwise_and(gray, inv_filtered_img)

            plt.imshow(img, cmap='gray')
            plt.show()

            # Binary Thresholding
            _, binary = cv2.threshold(img, 17, maxval=255, type=cv2.THRESH_BINARY)

            # Morphological opening
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Select largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Create mask from largest contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)
            img = cv2.bitwise_and(gray, mask)

            plt.imshow(img, cmap='gray')
            plt.show()

            # Applying Gamma correction and CLAHE
            img = apply_gamma(img)

            plt.imshow(img, cmap='gray')
            plt.show()
            img = apply_clahe(img)
            plt.imshow(img, cmap='gray')
            plt.show()

            return img

        # ----------------- Saving Full mammogram images -----------------

        print(f"Processing and saving {image_type} train_set images...")
        for idx, row in train_set.iterrows():
            overlay_img = process_img(
                img_path=row['base_image_path'], 
                augment=True
            )
            if overlay_img is None:
                print("Overlay image is none")
                continue
            else:
                label = 'benign' if row['label'] == 0 else 'malignant'
                filename = f"{row['case_id']}_{row['abnormality_type']}_{idx}_overlay.jpg"
                output_path = os.path.join(f"../../archive/{image_type}/train/{label}/" + filename)
                try: 
                    cv2.imwrite(output_path, overlay_img)
                except cv2.error as e:
                    print(f"Error saving overlay image for {row['case_id']}: {e}")
                    continue

        print(f"Processing and saving {image_type} val_set images...")
        for idx, row in val_set.iterrows():
            overlay_img = process_img(
                img_path=row['base_image_path'], 
                augment=True,
            )
            if overlay_img is None:
                continue
            else:
                filename = f"{row['case_id']}_{row['abnormality_type']}_{idx}_overlay.jpg"
                output_path = os.path.join(f"../../archive/{image_type}/validation/{label}/" + filename)
                try: 
                    cv2.imwrite(output_path, overlay_img)
                except cv2.error as e:
                    print(f"Error saving overlay image for {row['case_id']}: {e}")
                    continue

        print(f"Saving {image_type} test_set images...")
        for idx, row in test_set.iterrows():
            img = process_img(
                img_path="../../archive" + row['image_path'],
                augment=False
            )
            filename = f"{row['patient_id']}_{row['abnormality_type']}_{idx}_test.jpg"
            output_path = os.path.join(f"../../archive/{image_type}/test/{label}/" + filename)
            try: 
                cv2.imwrite(output_path, img)
            except cv2.error as e:
                print(f"Error saving overlay image for {row['patient_id']}: {e}")
                continue

        # ----------------- Saving Cropped images -----------------

        print("Processing and saving train_set images...")
        for idx, row in train_set.iterrows():
            if image_type == 'cropped images':
                to_save_img = process_cropped_img(
                    img_path=row['base_image_path']
                )
            else:
                to_save_img = process_img(
                    img_path=row['base_image_path'], 
                )
            
            if to_save_img is None:
                continue
            else:
                label = 'benign' if row['label'] == 0 else 'malignant'
                filename = f"{row['patient_id']}_{row['abnormality_type']}_{idx}_train.jpg"
                output_path = f"../../archive/5_classes/{image_type}/train/{label}_{row['abnormality_type']}/{filename}"
                try:
                    cv2.imwrite(output_path, to_save_img)
                except cv2.error as e:
                    print(f"Error saving training image for {row['patient_id']}: {e}")
                    continue

        print("Processing and saving val_set images...")
        for idx, row in val_set.iterrows():
            if image_type == 'cropped images':
                to_save_img = process_cropped_img(
                    img_path=row['base_image_path']
                )
            else:
                to_save_img = process_img(
                    img_path=row['base_image_path'], 
                )
            
            if to_save_img is None:
                continue
            else:
                label = 'benign' if row['label'] == 0 else 'malignant'
                filename = f"{row['patient_id']}_{row['abnormality_type']}_{idx}_validation.jpg"
                output_path = f"../../archive/5_classes/{image_type}/validation/{label}_{row['abnormality_type']}/{filename}"
                try:
                    cv2.imwrite(output_path, to_save_img)
                except cv2.error as e:
                    print(f"Error saving validation image for {row['patient_id']}: {e}")
                    continue

        print("Saving test set images")
        for idx, row in test_set.iterrows():
            to_save_img = cv2.imread(f"../../archive/{row['image_path']}",cv2.IMREAD_GRAYSCALE)
            if to_save_img is None:
                continue
            else:
                label = 'benign' if row['label'] == 0 else 'malignant'
                filename = f"{row['patient_id']}_{row['abnormality_type']}_{idx}_test.jpg"
                output_path = f"../../archive/5_classes/{image_type}/test/{label}_{row['abnormality_type']}/{filename}"
                try:
                    cv2.imwrite(output_path, to_save_img)
                except cv2.error as e:
                    print(f"Error saving test image for {row['patient_id']}: {e}")
                    continue

    # ----------------------------------------------------------------
    # ----------------------- Helper functions -----------------------
    # ----------------------------------------------------------------
    def parse_patient_id(self, patient_id):
        if not patient_id:
            return None
        patterns = [
            r'^(Mass|Calc)-(Training|Test)_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)(_\d+)?$',
            r'^P_(\d+)_(LEFT|RIGHT)_(CC|MLO)(_\d+)?(\.dcm)?$'
        ]

        for pattern in patterns:
            match = re.match(pattern, patient_id)
            if match:
                if 'Mass' in patient_id or 'Calc' in patient_id:
                    return {
                        'abnormality_type': match.group(1).lower(),
                        'dataset': match.group(2).lower(),
                        'patient_num': match.group(3),
                        'laterality': match.group(4),
                        'view': match.group(5),
                        'abnormality_id': match.group(6)[1:] if match.group(6) else '1'
                    }
                else: 
                    return {
                        'abnormality_type': None,
                        'dataset': None,
                        'patient_num': match.group(1),
                        'laterality': match.group(2),
                        'view': match.group(3),
                        'abnormality_id': match.group(4)[1:] if match.group(4) else '1'
                    }
        return None