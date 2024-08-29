import os
import shutil
import numpy as np
import nibabel as nib
import random
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

base_dataset_path = r"C:\BraTS2021_TrainingSet"

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

output_path = r"C:\brats_processed_cropped"
balanced_output_path = r"C:\balanced_dataset"
sets = ['train', 'val', 'test']
categories = ['tumour', 'non_tumour']
views = ['top', 'side', 'front']
random_seed = 42
random.seed(random_seed)

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

def is_informative_slice(slice_data, content_threshold=0.25, edge_threshold=0.16):
    non_black_pixels = np.sum(slice_data > 10)
    total_pixels = slice_data.size
    content_ratio = non_black_pixels / total_pixels

    edges = np.abs(np.diff(slice_data))
    edge_ratio = np.sum(edges > 10) / edges.size

    return (content_ratio > content_threshold) and (edge_ratio > edge_threshold)

def has_sufficient_variance(slice_data, var_threshold=100):
    return np.var(slice_data) > var_threshold

def save_slice(img, patient, view, slice_num, has_tumour, set_name):
    category = 'tumour' if has_tumour else 'non_tumour'
    save_path = os.path.join(output_path, set_name, category, f'{patient}_{view}_slice_{slice_num:03d}.png')

    width, height = img.size
    left = (width - 200) / 2
    top = (height - 200) / 2
    right = (width + 200) / 2
    bottom = (height + 200) / 2

    img = img.crop((left, top, right, bottom))

    if view in ['front', 'side']:
        img = img.rotate(90, expand=True)

    img.save(save_path)

def process_patient_data(patient_folder, set_name): #Process data for a single patient by extractn
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

        slice_window = 5  # Save one slice every 5 slices
        for i in tqdm(range(0, slices, slice_window), desc=f"Processing {patient} - {view}", leave=False):
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
                flair_slice = (flair_normalized * 255).astype(np.uint8)

                img = Image.fromarray(flair_slice)
                has_tumour = np.any(seg_slice > 0)
                save_slice(img, patient, view, i, has_tumour, set_name)

def undersample_majority_class(src_dir, dest_dir, random_state):
    # Get the number of images in each class
    tumour_path = os.path.join(src_dir, 'tumour')
    non_tumour_path = os.path.join(src_dir, 'non_tumour')
    tumour_images = os.listdir(tumour_path)
    non_tumour_images = os.listdir(non_tumour_path)
    n_keep = min(len(tumour_images), len(non_tumour_images))
    random.seed(random_state)
    
    # Randomly select images from the majority class
    if len(tumour_images) > len(non_tumour_images):
        tumour_images = random.sample(tumour_images, n_keep)
    else:
        non_tumour_images = random.sample(non_tumour_images, n_keep)
    
    os.makedirs(os.path.join(dest_dir, 'tumour'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'non_tumour'), exist_ok=True)
    
    # Copy selected images to destination
    for img in tumour_images:
        shutil.copy(os.path.join(tumour_path, img), os.path.join(dest_dir, 'tumour', img))
    for img in non_tumour_images:
        shutil.copy(os.path.join(non_tumour_path, img), os.path.join(dest_dir, 'non_tumour', img))

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

    for set_name, folders in [('train', train), ('val', val), ('test', test)]:
        for patient_folder in tqdm(folders, desc=f"Processing {set_name} set"):
            patient_path = os.path.join(base_dataset_path, patient_folder)
            process_patient_data(patient_path, set_name)
   
    print("Starting undersampling.....")
    for subset in sets:
        src_dir = os.path.join(output_path, subset)
        dest_dir = os.path.join(balanced_output_path, subset)
        undersample_majority_class(src_dir, dest_dir, random_state=random_seed)
    
    print("Preprocessing complete!")