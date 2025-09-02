import os
import random
import shutil
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


random.seed(42)

def create_directories_and_split_data():

    cwd=os.getcwd()

    # If cwd doesn't end with "Plant_Disease_Identifier", assume that the project root is the parent folder
    if os.path.basename(cwd).lower() != "Plant_Disease_Identifier":
        PROJECT_ROOT = os.path.abspath(os.path.join(cwd, ".."))
    else:
        PROJECT_ROOT = cwd
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR,exist_ok=True)
    DATA_DIR=os.path.join(DATA_DIR,"PlantVillage")
    os.makedirs(DATA_DIR,exist_ok=True)
    TRAIN_DIR=os.path.join(DATA_DIR,"train")
    VAL_DIR=os.path.join(DATA_DIR,"val")
    TEST_DIR=os.path.join(DATA_DIR,"test")
    # Create the "train" and "test" directories if they don't exist yet
    os.makedirs(TRAIN_DIR,exist_ok=True)
    os.makedirs(VAL_DIR,exist_ok=True)
    os.makedirs(TEST_DIR,exist_ok=True)

    CLASS_NAMES=[d for d in os.listdir(DATA_DIR) if (os.path.isdir(os.path.join(DATA_DIR, d)) and d not in ["train","val","test"])]

    for i in CLASS_NAMES:
        TRAIN_DIR_PATH=os.path.join(TRAIN_DIR,i)
        VAL_DIR_PATH=os.path.join(VAL_DIR,i)
        TEST_DIR_PATH=os.path.join(TEST_DIR,i)
        os.makedirs(TRAIN_DIR_PATH,exist_ok=True)
        os.makedirs(VAL_DIR_PATH,exist_ok=True)
        os.makedirs(TEST_DIR_PATH,exist_ok=True)



    SPLIT_RATIO=0.15
    for i in CLASS_NAMES:
        SOURCE_DIR=os.path.join(DATA_DIR,i)
        all_files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
        random.seed(42)
        random.shuffle(all_files)
        num_val_files = int(len(all_files)*SPLIT_RATIO)
        num_test_files=num_val_files
        val_files=all_files[:num_val_files]
        test_files=all_files[num_val_files:num_val_files+num_test_files]
        train_files=all_files[num_val_files+num_test_files:]
        for file_name in train_files:
            SOURCE_PATH=os.path.join(SOURCE_DIR,file_name)
            DESTINATION_PATH=os.path.join(TRAIN_DIR,i,file_name)
            shutil.move(SOURCE_PATH,DESTINATION_PATH)
        for file_name in val_files:
            SOURCE_PATH=os.path.join(SOURCE_DIR,file_name)
            DESTINATION_PATH=os.path.join(VAL_DIR,i,file_name)
            shutil.move(SOURCE_PATH,DESTINATION_PATH)
        for file_name in test_files:
            SOURCE_PATH=os.path.join(SOURCE_DIR,file_name)
            DESTINATION_PATH=os.path.join(TEST_DIR,i,file_name)
            shutil.move(SOURCE_PATH,DESTINATION_PATH)



def get_dataloader(ROOT_DIR,TRANSFORMATION,BATCH_SIZE,shuffle_bool_value,NUM_WORKERS=0):
    InitialData=datasets.ImageFolder(root=ROOT_DIR,transform=TRANSFORMATION)
    dataloader=DataLoader(InitialData,batch_size=BATCH_SIZE,shuffle=shuffle_bool_value,num_workers=NUM_WORKERS)
    return dataloader

def image_transformation_EfficientNetB1_DataAugmentation(inference_transforms):       # inference_transform is the auto_transform returned by the EfficientNet-B1 model
    train_transforms=transforms.Compose([
            transforms.RandomResizedCrop(240, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=inference_transforms.mean, std=inference_transforms.std) # Normalize the tensor
        ])
    return train_transforms