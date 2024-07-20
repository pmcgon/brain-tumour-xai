import os
import shutil
import json
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

base_dataset_path = r"D:\Downlaods\BraTS2021_TrainingSet"

# List of all dataset folders
dataset_folders = [
    "ACRIN-FMISO-Brain",
    "CPTAC-GBM",
    "IvyGAP",
    "new-not-previously-in-TCIA",
    "TCGA-GBM",
    "TCGA-LGG",
    "UCSF-PDGM",
    "UPENN-GBM"
]

output_path = r"D:\brats_processed"
# Define constants
sets = ['train', 'val', 'test']
categories = ['tumor', 'non_tumor']
views = ['axial', 'sagittal', 'coronal']

def create_directory_structure():
    for set_name in sets:
        for category in categories:
            path = os.path.join(output_path, set_name, category)
            os.makedirs(path, exist_ok=True)

def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def is_informative_slice(slice_data, threshold=0.25):
    non_black_pixels = np.sum(slice_data > 10)
    total_pixels = slice_data.size
    return (non_black_pixels / total_pixels) > threshold

def save_slice(img, patient_name, view, slice_num, has_tumor, set_name):
    category = 'tumor' if has_tumor else 'non_tumor'
    save_path = os.path.join(output_path, set_name, category, f'{patient_name}_{view}_slice_{slice_num:03d}.png')

    # Pad the image to 240x240 if it's not already that size
    if img.size != (240, 240):
        img = pad_image_to_240x240(img)

    # Rotate coronal and sagittal views 90 degrees to the left
    if view in ['coronal', 'sagittal']:
        img = img.rotate(90, expand=True)

    img.save(save_path)
def pad_image_to_240x240(img):
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # Get current dimensions
        height, width = img_array.shape
        
        # Calculate padding
        pad_height = max(240 - height, 0)
        pad_width = max(240 - width, 0)
        
        # Pad the image
        padded_img = np.pad(img_array, 
                            ((pad_height//2, pad_height - pad_height//2), 
                            (pad_width//2, pad_width - pad_width//2)),
                            mode='constant', constant_values=0)
        
        # If the padded image is larger than 240x240, crop it
        if padded_img.shape[0] > 240 or padded_img.shape[1] > 240:
            padded_img = padded_img[:240, :240]
                    
        return Image.fromarray(padded_img)


def process_patient_data(patient_folder, set_name):
    seg_file = [f for f in os.listdir(patient_folder) if f.endswith('seg.nii.gz')][0]
    seg_path = os.path.join(patient_folder, seg_file)
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    t1ce_file = [f for f in os.listdir(patient_folder) if f.endswith('flair.nii.gz')][0]
    t1ce_path = os.path.join(patient_folder, t1ce_file)
    t1ce_img = nib.load(t1ce_path)
    t1ce_data = t1ce_img.get_fdata()

    patient_name = os.path.basename(patient_folder)

    for view_idx, view in enumerate(views):
        if view == 'axial':
            slices = t1ce_data.shape[2]
        elif view == 'sagittal':
            slices = t1ce_data.shape[0]
        else:  # coronal
            slices = t1ce_data.shape[1]

        for i in tqdm(range(slices), desc=f"Processing {patient_name} - {view}", leave=False):
            if view == 'axial':
                t1ce_slice = t1ce_data[:, :, i]
                seg_slice = seg_data[:, :, i]
            elif view == 'sagittal':
                t1ce_slice = t1ce_data[i, :, :]
                seg_slice = seg_data[i, :, :]
            else:  # coronal
                t1ce_slice = t1ce_data[:, i, :]
                seg_slice = seg_data[:, i, :]

            # Handle division by zero and NaN values
            t1ce_min = np.min(t1ce_slice)
            t1ce_max = np.max(t1ce_slice)
            if t1ce_min == t1ce_max:
                t1ce_normalized = np.zeros_like(t1ce_slice)
            else:
                t1ce_normalized = (t1ce_slice - t1ce_min) / (t1ce_max - t1ce_min)
            
            t1ce_normalized = np.clip(t1ce_normalized, 0, 1)  # Ensure values are between 0 and 1
            t1ce_slice = (t1ce_normalized * 255).astype(np.uint8)

            if is_informative_slice(t1ce_slice):
                img = Image.fromarray(t1ce_slice)
                has_tumor = np.any(seg_slice > 0)
                save_slice(img, patient_name, view, i, has_tumor, set_name)

    
    

# Main execution
if __name__ == "__main__":
    # Create and clear output directories
    create_directory_structure()
    for set_name in sets:
        for category in categories:
            clear_directory(os.path.join(output_path, set_name, category))

    print("Output directories cleared and ready.")

    # Get list of all patient folders across all dataset folders
    all_patient_folders = []
    for folder in dataset_folders:
        folder_path = os.path.join(base_dataset_path, folder)
        if os.path.exists(folder_path):
            patient_folders = [os.path.join(folder, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            all_patient_folders.extend(patient_folders)
        else:
            print(f"Warning: Folder {folder} not found in {base_dataset_path}")

    # Split into train, val, and test sets
    train_val, test = train_test_split(all_patient_folders, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)  # 0.1765 of 85% is 15% of total

    # Save the split information
    split_info = {
        'train': train,
        'val': val,
        'test': test
    }
    with open(os.path.join(output_path, 'data_split.json'), 'w') as f:
        json.dump(split_info, f)

    # Process each set
    for set_name, folders in [('train', train), ('val', val), ('test', test)]:
        for patient_folder in tqdm(folders, desc=f"Processing {set_name} set"):
            patient_path = os.path.join(base_dataset_path, patient_folder)
            process_patient_data(patient_path, set_name)

    print("Processing complete!")
