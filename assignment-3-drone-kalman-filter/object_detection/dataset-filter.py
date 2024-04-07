import os

def filter_lines(input_folder):
    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        
        # Check if the file is a text file
        if filename.endswith(".txt"):
            lines_to_keep = []
            found_valid_line = False
            
            # Read the file line by line
            with open(filepath, "r") as file:
                for line in file:
                    # Check if the line starts with 0, 1, 2, or 3
                    if line.strip() and line.strip()[0] in ['0', '1', '2','3']:
                        lines_to_keep.append(line)
                        found_valid_line = True
            
            # Check if any lines to keep were found
            if found_valid_line:
                # Write the filtered lines back to the file
                with open(filepath, "w") as file:
                    file.writelines(lines_to_keep)
            else:
                # No lines to keep were found, delete the file
                os.remove(filepath)

# Main
input_folder = "2-obj-detection-model\datasets-custom\VisDrone\VisDrone2019-DET-train\labels"
filter_lines(input_folder)