def filter_correct_lines(file_path, output_file_path, correct_value_count=2782):
    try:
        with open(file_path, 'r') as file:
            with open(output_file_path, 'w') as output_file:
                line_number = 1
                for line in file:
                    # Strip whitespace and newline characters, then split by comma
                    values = line.strip().split(',')
                    # Count the number of values
                    num_values = len(values)
                    if num_values == correct_value_count:
                        output_file.write(line)  # Write the correct line to the new file
                    else:
                        print(f"Line {line_number}: {num_values} values (Incorrect line)")
                    line_number += 1
                print("Total number of lines checked:", line_number - 1)
                print(f"Correct lines written to: {output_file_path}")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
file_path = 'data_file.txt'  # Replace with the path to your input file
output_file_path = 'filtered_data_file.txt'  # Replace with the path for your output file
filter_correct_lines(file_path, output_file_path)
