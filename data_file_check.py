def count_values_per_line(file_path):
    try:
        with open(file_path, 'r') as file:
            line_number = 1
            for line in file:
                # Strip whitespace and newline characters, then split by comma
                values = line.strip().split(',')
                # Count the number of values
                num_values = len(values)
                if num_values != 2782:
                    print(f"Line {line_number}: {num_values} values")
                line_number += 1
            print("Total number of lines checked:", line_number - 1)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
file_path = 'data_file.txt'  # Replace with the path to your file
count_values_per_line(file_path)
