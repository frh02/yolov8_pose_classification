import os

def change_file_name(folder_path,name_of_file=str):

  # Verify that the folder path is valid
  if not os.path.exists(folder_path):
      print(f"The folder '{folder_path}' does not exist.")
      exit()

  # List all files in the folder
  files = os.listdir(folder_path)

  # Iterate through the files and rename them
  for i, file_name in enumerate(files, start=1):
      # Create the new file name
      new_file_name = f'{name_of_file}_{i}.jpg'
      
      # Construct the full paths for the old and new names
      old_path = os.path.join(folder_path, file_name)
      new_path = os.path.join(folder_path, new_file_name)
      
      # Rename the file
      os.rename(old_path, new_path)
      
      print(f'Renamed: {file_name} to {new_file_name}')

  print('Renaming completed.')


folder_path = input('Enter the folder path: ')
name_of_file = input('Enter the new file name: ')
change_file_name(folder_path,name_of_file)