import os

def rename_files(folder_path):
    # Get all files in the folder
    files = os.listdir(folder_path)
    # Sort files to ensure renaming in a progressive order
    files.sort()

    # Rename each file iteratively
    for i, file_name in enumerate(files):
        # Define the new file name based on the current index
        new_file_name = f"sit_front_8_{i+1}.jpg"  # You can adjust the naming pattern as per your requirement
        
        # Construct the full paths for old and new file names
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{file_name}' to '{new_file_name}'")

# Replace 'folder_path' with the path to your folder containing JPG files
folder_path = 'Data\Training_STS\Front\Sitting\Sitting 2-samples'
rename_files(folder_path)
