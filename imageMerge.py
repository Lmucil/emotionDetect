import os
import shutil

original_path = "dataset_FER_2013/train"   
augmented_path = "CleansingData/train"     

print("merging " + original_path + " with " + augmented_path )
for emotion in os.listdir(augmented_path):
    aug_emotion_folder = os.path.join(augmented_path, emotion)
    orig_emotion_folder = os.path.join(original_path, emotion)

    for file in os.listdir(aug_emotion_folder):
        src = os.path.join(aug_emotion_folder, file)
        dest = os.path.join(orig_emotion_folder, file)
        shutil.move(src, dest) 

    print(orig_emotion_folder+ ": Merged")

