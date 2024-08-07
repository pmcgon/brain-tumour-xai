import os
import shutil
import json
import numpy as np
import nibabel as nib
import random
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

base_dataset_path = r"D:\BraTS2021_TrainingSet"
output_path = r"D:\brats_processed_pretrained"
sets = ['train', 'val', 'test']
categories = ['tumour', 'non_tumour']
views = ['top', 'side', 'front']
random.seed(42)

# This variable was missing in the original code
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

def is_informative_slice(slice_data, content_threshold=0.25, edge_threshold=0.1):
    non_black_pixels = np.sum(slice_data > 0.1)
    total_pixels = slice_data.size
    content_ratio = non_black_pixels / total_pixels

    edges = np.abs(np.diff(slice_data))
    edge_ratio = np.sum(edges > 0.1) / edges.size

    return (content_ratio > content_threshold) and (edge_ratio > edge_threshold)

def has_sufficient_variance(slice_data, var_threshold=0.01):
    return np.var(slice_data) > var_threshold

def save_slice(img, patient, view, slice_num, has_tumour, set_name):
    category = 'tumour' if has_tumour else 'non_tumour'
    save_path = os.path.join(output_path, set_name, category, f'{patient}_{view}_slice_{slice_num:03d}.png')

    # Resize to 224x224
    img = img.resize((224, 224), Image.LANCZOS)

    # Convert to RGB
    img = img.convert('RGB')

    if view in ['front', 'side']:
        img = img.rotate(90, expand=True)

    img.save(save_path)

def shuffle_data(data_list):
    random.shuffle(data_list)
    return data_list

def process_patient_data(patient_folder, set_name):
    seg_file = [f for f in os.listdir(patient_folder) if f.endswith('seg.nii.gz')][0]
    seg_path = os.path.join(patient_folder, seg_file)
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    flair_file = [f for f in os.listdir(patient_folder) if f.endswith('flair.nii.gz')][0]
    flair_path = os.path.join(patient_folder, flair_file)
    flair_img = nib.load(flair_path)
    flair_data = flair_img.get_fdata()

    patient = os.path.basename(patient_folder)

    for view in views:
        if view == 'top':
            slices = flair_data.shape[2]
        elif view == 'front':
            slices = flair_data.shape[1]
        else:  # side
            slices = flair_data.shape[0]

        slice_window = 9 # Save one slice every 5 slices
        for i in range(0, slices, slice_window):
            if view == 'top':
                flair_slice = flair_data[:, :, i]
                seg_slice = seg_data[:, :, i]
            elif view == 'side':
                flair_slice = flair_data[i, :, :]
                seg_slice = seg_data[i, :, :]
            else:  # front
                flair_slice = flair_data[:, i, :]
                seg_slice = seg_data[:, i, :]

            if is_informative_slice(flair_slice) and has_sufficient_variance(flair_slice):
                flair_min = np.min(flair_slice)
                flair_max = np.max(flair_slice)
                if flair_min == flair_max:
                    flair_normalized = np.zeros_like(flair_slice)
                else:
                    flair_normalized = (flair_slice - flair_min) / (flair_max - flair_min)
                
                flair_normalized = np.clip(flair_normalized, 0, 1)

                img = Image.fromarray((flair_normalized * 255).astype(np.uint8))
                has_tumour = np.any(seg_slice > 0)
                save_slice(img, patient, view, i, has_tumour, set_name)

def count_images():
    counts = {set_name: {category: 0 for category in categories} for set_name in sets}
    for set_name in sets:
        for category in categories:
            path = os.path.join(output_path, set_name, category)
            counts[set_name][category] = len([f for f in os.listdir(path) if f.endswith('.png')])
    return counts

# Main execution
if __name__ == "__main__":
    create_directory_structure()
    for set_name in sets:
        for category in categories:
            clear_directory(os.path.join(output_path, set_name, category))

    print("Existing directories cleared")

    all_patient_folders = []
    for folder in dataset_folders:
        folder_path = os.path.join(base_dataset_path, folder)
        if os.path.exists(folder_path):
            patient_folders = [os.path.join(folder, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            all_patient_folders.extend(patient_folders)
        else:
            print(f"Warning: Folder {folder} not found in {base_dataset_path}")

    train_val, test = train_test_split(all_patient_folders, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)

    train = shuffle_data(train)
    val = shuffle_data(val)
    test = shuffle_data(test)

    split_info = {
        'train': train,
        'val': val,
        'test': test
    }

    with open(os.path.join(output_path, 'data_split.json'), 'w') as f:
        json.dump(split_info, f)

    for set_name, folders in [('train', train), ('val', val), ('test', test)]:
        for patient_folder in tqdm(folders, desc=f"Processing {set_name} set"):
            patient_path = os.path.join(base_dataset_path, patient_folder)
            process_patient_data(patient_path, set_name)

    image_counts = count_images()
    print("Image counts:")
    for set_name in sets:
        print(f"{set_name}:")
        for category in categories:
            print(f"  {category}: {image_counts[set_name][category]}")

    print("Preprocessing complete!")
