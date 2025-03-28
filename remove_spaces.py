import os

def remove_spaces_from_files(directory):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory '{directory}' does not exist.")
        return
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(directory, filename)
            
            try:
                # Open the file and read its content with utf-8 encoding
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Remove all spaces
                content_no_spaces = content.replace(" ", "")
                
                # Write the modified content back to the file with utf-8 encoding
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content_no_spaces)
                
                print(f"Spaces removed from {filename}")

            except UnicodeDecodeError:
                print(f"Could not decode the file {filename} with utf-8 encoding. Skipping this file.")
            except Exception as e:
                print(f"An error occurred with file {filename}: {e}")

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    remove_spaces_from_files(directory)
