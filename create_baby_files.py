#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

def create_baby_files(source_dir, target_dir, max_lines=1000):
    """
    For each text file in source_dir, create a new file named [original_name]baby.txt
    in the target_dir containing only the first max_lines lines.
    
    Args:
        source_dir (str): Directory containing the original text files
        target_dir (str): Directory where baby files will be created
        max_lines (int): Maximum number of lines to include in baby files
    """
    # Convert to Path objects for easier handling
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    processed = 0
    skipped = 0
    
    # Process all files in the source directory
    for file_path in source_path.iterdir():
        # Only process files (not directories) and only text files
        if file_path.is_file() and file_path.suffix.lower() == '.txt':
            # Create the baby filename
            original_name = file_path.stem  # Get filename without extension
            baby_filename = f"{original_name}baby.txt"
            target_file = target_path / baby_filename
            
            try:
                # Read first max_lines lines from the source file
                with open(file_path, 'r', encoding='utf-8') as src_file:
                    lines = []
                    for i, line in enumerate(src_file):
                        if i >= max_lines:
                            break
                        lines.append(line)
                
                # Write the lines to the baby file
                with open(target_file, 'w', encoding='utf-8') as tgt_file:
                    tgt_file.writelines(lines)
                
                print(f"Created: {baby_filename} ({len(lines)} lines)")
                processed += 1
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                skipped += 1
        else:
            # Skip non-text files
            if file_path.is_file():
                print(f"Skipping non-text file: {file_path.name}")
                skipped += 1
    
    print(f"\nSummary:")
    print(f"  Files processed: {processed}")
    print(f"  Files skipped: {skipped}")
    print(f"  Baby files created in: {target_path.absolute()}")
    print(f"  Each baby file contains up to {max_lines} lines")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a baby version of each text file containing only the first 1000 lines."
    )
    parser.add_argument(
        "source_dir", 
        help="Directory containing the original text files"
    )
    parser.add_argument(
        "target_dir", 
        help="Directory where baby files will be created"
    )
    parser.add_argument(
        "--max-lines", 
        type=int, 
        default=1000,
        help="Maximum number of lines to include in baby files (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Run the main function
    create_baby_files(args.source_dir, args.target_dir, args.max_lines)

if __name__ == "__main__":
    main()