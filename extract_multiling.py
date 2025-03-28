#!/usr/bin/env python3
import json
import argparse
import os

def extract_text_from_jsonlines(jsonl_file_path, output_file_path):
    """
    Extract all values from "text" keys in a JSONL file and combine them into a single text file.
    Each line in the input file is expected to be a complete JSON object with a "text" field.
    Unicode escape sequences (like \u0623) will be properly converted to their actual characters.
    
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
                        # json.loads() automatically converts Unicode escape sequences to characters
                        data = json.loads(line)
                        
                        # Extract the text value if present
                        if "text" in data and isinstance(data["text"], str):
                            text_values.append(data["text"])
                        else:
                            print(f"Warning: Line {line_num} does not contain a valid 'text' field")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON object at line {line_num}: {line[:50]}... Error: {e}")
        
        if not text_values:
            print("No 'text' fields found in the JSONL file.")
            return
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Write the combined text to the output file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for text in text_values:
                file.write(text + "\n\n")  # Add empty line between entries for readability
        
        print(f"Successfully extracted {len(text_values)} text entries to {output_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File {jsonl_file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def extract_text_without_extension_check(jsonl_file_path, output_file_path):
    """
    Same as extract_text_from_jsonlines but without checking file extension.
    Used when the --force flag is provided.
    """
    # This is a simple wrapper that calls the main function but bypasses the extension check
    try:
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
                        else:
                            print(f"Warning: Line {line_num} does not contain a valid 'text' field")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON object at line {line_num}: {line[:50]}... Error: {e}")
        
        if not text_values:
            print("No 'text' fields found in the JSONL file.")
            return
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
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
    parser = argparse.ArgumentParser(description="Extract and combine 'text' values from a JSONL file with proper Unicode handling")
    parser.add_argument("jsonl_file", help="Path to the JSONL file")
    parser.add_argument("output_file", help="Path to save the output text file")
    parser.add_argument("--force", action="store_true", help="Process the file regardless of extension")
    parser.add_argument("--encoding", default="utf-8", help="Encoding to use for input and output files (default: utf-8)")
    
    args = parser.parse_args()
    
    # If --force is used, call the function that bypasses extension check
    if args.force:
        extract_text_without_extension_check(args.jsonl_file, args.output_file)
    else:
        extract_text_from_jsonlines(args.jsonl_file, args.output_file)