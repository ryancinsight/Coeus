import json
import sys

def parse_errors():
    errors = []
    try:
        with open('check_output.json', 'r') as f:
            for line in f:
                try:
                    msg = json.loads(line)
                    if msg.get('reason') == 'compiler-message':
                        message = msg.get('message', {})
                        if message.get('level') == 'error':
                            spans = message.get('spans', [])
                            if spans and ('nn\\' in spans[0].get('file_name') or 'nn/' in spans[0].get('file_name')):
                                errors.append(message)
                except:
                    continue
    except FileNotFoundError:
        print("check_output.json not found")
        return

    print(f"Found {len(errors)} errors in nn crate")
    for i, err in enumerate(errors[:20]):
        print(f"Error {i+1}:")
        print(f"  Message: {err.get('message')}")
        spans = err.get('spans', [])
        if spans:
            print(f"  File: {spans[0].get('file_name')}:{spans[0].get('line_start')}")
        print("-" * 40)

if __name__ == "__main__":
    parse_errors()
