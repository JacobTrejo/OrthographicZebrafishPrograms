import os

def get_base_filenames_in_directory(directory):
    """Get a set of base filenames (without extensions) in the given directory."""
    return {os.path.splitext(file)[0] for file in os.listdir(directory)}

def find_unique_files(dir1, dir2):
    """Find files that are not common between two directories, ignoring extensions."""
    base_files_dir1 = get_base_filenames_in_directory(dir1)
    base_files_dir2 = get_base_filenames_in_directory(dir2)

    # Find unique base filenames in each directory
    unique_to_dir1 = base_files_dir1 - base_files_dir2
    unique_to_dir2 = base_files_dir2 - base_files_dir1

    return unique_to_dir1, unique_to_dir2

# Specify the directories
directory2 = 'resnet_labels/train_compressed_4/'
directory1 = 'images/train/'

# Find unique files
unique_files_dir1, unique_files_dir2 = find_unique_files(directory1, directory2)

print(f"Files unique to {directory1} (ignoring extensions):")
for file in unique_files_dir1:
    print(file)

print(f"\nFiles unique to {directory2} (ignoring extensions):")
for file in unique_files_dir2:
    print(file)
    os.remove(f'{directory2}{file}.pt')
    print(f'Deleted {file}')

