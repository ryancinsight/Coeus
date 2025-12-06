import json
import sys
import os

def parse_errors():
    errors = []
    try:
        with open('test_output_2.json', 'r') as f:
            for line in f:
                try:
                    msg = json.loads(line)
                    if msg.get('reason') == 'compiler-message':
                        message = msg.get('message', {})
                        if message.get('level') == 'error':
                            errors.append(message)
                except:
                    continue
    except FileNotFoundError:
        print("test_output_2.json not found")
        return

    print(f"Found {len(errors)} errors in nn tests")
    
    # Group by message to find common patterns
    msg_counts = {}
    file_errors = {}
    
    for err in errors:
        msg_text = err.get('message')
        if msg_text not in msg_counts:
            msg_counts[msg_text] = []
        msg_counts[msg_text].append(err)
        
        spans = err.get('spans', [])
        if spans:
            file_name = spans[0].get('file_name')
            if file_name not in file_errors:
                file_errors[file_name] = []
            file_errors[file_name].append(err)
        
    print("Top 5 error messages:")
    sorted_msgs = sorted(msg_counts.items(), key=lambda x: len(x[1]), reverse=True)
    for msg, errs in sorted_msgs[:5]:
        print(f"  {len(errs)}x: {msg}")
        # Print one example location
        spans = errs[0].get('spans', [])
        if spans:
            print(f"     Example: {spans[0].get('file_name')}:{spans[0].get('line_start')}")
        print("-" * 20)
        
    print("\nTop 5 affected files:")
    sorted_files = sorted(file_errors.items(), key=lambda x: len(x[1]), reverse=True)
    for fname, errs in sorted_files[:5]:
        print(f"  {len(errs)}x: {fname}")

if __name__ == "__main__":
    parse_errors()
