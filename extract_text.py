import json
import argparse
import os

def extract_text_from_jsonlines(jsonl_file_path, output_file_path):
    """
    Extract all values from "text" keys in a JSONL file and combine them into a single text file.
    Each line in the input file is expected to be a complete JSON object with a "text" field.
    
    Args:
        jsonl_file_path (str): Path to the JSONL file
        output_file_path (str): Path to save the output text file
    """
    try:
        # Check if the file has a .jsonl extension
        file_ext = os.path.splitext(jsonl_file_path)[1].lower()
        if file_ext != '.jsonl':
            print(f"Warning: Input file '{jsonl_file_path}' does not have a .jsonl extension.")
            user_continue = input("Continue anyway? (y/n): ")
            if user_continue.lower() != 'y':
                print("Operation canceled.")
                return
        
        text_values = []
        
        # Read the JSONL file (one JSON object per line)
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line.strip():  # Skip empty lines
                    try:
                        # Parse each line as a separate JSON object
                        data = json.loads(line)
                        
                        # Extract the text value if present
                        if "text" in data and isinstance(data["text"], str):
                            text_values.append(data["text"])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON object at line {line_num}: {line[:50]}... Error: {e}")
        
        if not text_values:
            print("No 'text' fields found in the JSONL file.")
            return
            
        # Write the combined text to the output file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for text in text_values:
                file.write(text + "\n\n")  # Add empty line between entries for readability
        
        print(f"Successfully extracted {len(text_values)} text entries to {output_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File {jsonl_file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and combine 'text' values from a JSONL file")
    parser.add_argument("jsonl_file", help="Path to the JSONL file")
    parser.add_argument("output_file", help="Path to save the output text file")
    parser.add_argument("--force", action="store_true", help="Process the file regardless of extension")
    
    args = parser.parse_args()
    
    # If --force is used, rename the function to bypass extension check
    if args.force:
        extract_text_from_jsonlines_no_check = extract_text_from_jsonlines.__wrapped__ if hasattr(extract_text_from_jsonlines, "__wrapped__") else extract_text_from_jsonlines
        extract_text_from_jsonlines_no_check(args.jsonl_file, args.output_file)
    else:
        extract_text_from_jsonlines(args.jsonl_file, args.output_file)