#!/usr/bin/env python3
import argparse
from pathlib import Path

def calculate_accuracy(predictions_file, answers_file):
    """
    Calculate accuracy by comparing predictions against answers.
    
    A prediction is considered correct if any of the characters in a prediction line
    (up to 3 characters, including spaces) matches the single character in the
    corresponding answer line.
    
    If an answer line is empty, the prediction is considered correct if it contains a space.
    
    Args:
        predictions_file (str): Path to the predictions file, where each line has up to 3 characters
        answers_file (str): Path to the answers file, where each line has 1 character or is empty
        
    Returns:
        tuple: (correct_lines, total_lines, accuracy, list of correct line numbers, list of incorrect line numbers)
    """
    try:
        # Read all lines from both files
        with open(predictions_file, 'r', encoding='utf-8') as pred_file:
            predictions = [line.rstrip('\n') for line in pred_file]  # Keep spaces but remove newlines
            
        with open(answers_file, 'r', encoding='utf-8') as ans_file:
            answers = [line.strip() for line in ans_file]  # Remove whitespace from answers
        
        # Remove any empty lines from the end of files
        while predictions and not predictions[-1]:
            predictions.pop()
        while answers and not answers[-1]:
            answers.pop()
            
        # Check if files have content
        if not predictions or not answers:
            raise ValueError("One or both input files are empty")
            
        # Check if the number of lines match
        if len(predictions) != len(answers):
            raise ValueError(f"Number of lines in predictions ({len(predictions)}) doesn't match answers ({len(answers)})")
        
        correct_lines = []
        incorrect_lines = []
        
        # Compare each prediction with corresponding answer
        for i, (pred, ans) in enumerate(zip(predictions, answers), 1):
            # Pad prediction to 3 characters if needed (with spaces)
            pred = pred.ljust(3)[:3]
            
            # Each prediction should ideally have 3 characters
            if len(pred) < 3:
                print(f"Warning: Line {i} in predictions has only {len(pred)} characters, expected 3: '{pred}'")
            
            # If answer is empty, check for space in prediction
            if not ans:
                if ' ' in pred:
                    correct_lines.append(i)
                else:
                    incorrect_lines.append(i)
                continue
                
            # Each answer should have 1 character
            if len(ans) != 1:
                print(f"Warning: Line {i} in answers has {len(ans)} characters, expected 1: '{ans}'")
                ans = ans[0] if ans else ""
            
            # Check if any character in prediction matches the answer
            if any(char == ans for char in pred):
                correct_lines.append(i)
            else:
                incorrect_lines.append(i)
        
        # Calculate accuracy
        total_lines = len(predictions)
        correct_count = len(correct_lines)
        accuracy = (correct_count / total_lines) * 100 if total_lines > 0 else 0
        
        return correct_count, total_lines, accuracy, correct_lines, incorrect_lines
        
    except Exception as e:
        print(f"Error processing files: {e}")
        return 0, 0, 0, [], []

def main():
    parser = argparse.ArgumentParser(description="Calculate prediction accuracy against answers")
    parser.add_argument("predictions_file", help="Path to the predictions file (each line has up to 3 characters)")
    parser.add_argument("answers_file", help="Path to the answers file (each line has 1 character or is empty)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output with line numbers")
    parser.add_argument("--show-mismatches", "-m", action="store_true", help="Show prediction vs answer for incorrect lines")
    
    args = parser.parse_args()
    
    # Calculate accuracy
    correct, total, accuracy, correct_lines, incorrect_lines = calculate_accuracy(
        args.predictions_file, args.answers_file
    )
    
    # Print results
    print(f"\nAccuracy Results:")
    print(f"----------------")
    print(f"Total predictions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if True:
        print("\nCorrect predictions on lines:")
        if correct_lines:
            for i, line_num in enumerate(correct_lines):
                if i > 0 and i % 10 == 0:
                    print()  # Line break every 10 items for readability
                print(f"{line_num}", end=", " if i < len(correct_lines)-1 else "")
            print()
        else:
            print("None")
        
        print("\nIncorrect predictions on lines:")
        if incorrect_lines:
            for i, line_num in enumerate(incorrect_lines):
                if i > 0 and i % 10 == 0:
                    print()  # Line break every 10 items for readability
                print(f"{line_num}", end=", " if i < len(incorrect_lines)-1 else "")
            print()
        else:
            print("None")
    
    if True:
        print("\nIncorrect Predictions Details:")
        print("-----------------------------")
        
        # Re-read files to show actual content
        with open(args.predictions_file, 'r', encoding='utf-8') as pred_file:
            predictions = [line.rstrip('\n') for line in pred_file]
            
        with open(args.answers_file, 'r', encoding='utf-8') as ans_file:
            answers = [line.strip() for line in ans_file]
        
        for line_num in incorrect_lines:
            pred = predictions[line_num-1].ljust(3)[:3]
            ans = answers[line_num-1]
            print(f"Line {line_num}: Prediction '{pred}' ≠ Answer '{ans}'")

if __name__ == "__main__":
    main()