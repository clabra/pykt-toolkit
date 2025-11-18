#!/usr/bin/env python3
"""
Fix indentation bug in gainakt3_exp.py

The mastery computation block (lines 459-769) has 12 spaces of indentation
because it was the body of a commented-out elif statement.
This script un-indents those lines by 4 spaces to restore correct structure.
"""

import sys

def fix_indentation(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Lines to un-indent (0-indexed, so subtract 1)
    start_line = 459 - 1  # Line 459 in editor = index 458
    end_line = 769 - 1    # Line 769 in editor = index 768
    
    # Line with orphaned else to remove
    orphaned_else_start = 770 - 1  # Line 770: "        else:"
    orphaned_else_end = 777 - 1    # Line 777: last line of else block
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if start_line <= i <= end_line:
            # Un-indent by 4 spaces (from 12 to 8)
            if line.startswith('            '):  # 12 spaces
                fixed_line = line[4:]  # Remove 4 spaces
                fixed_lines.append(fixed_line)
            else:
                # Line doesn't have 12-space indent (blank line, different indent)
                fixed_lines.append(line)
        elif orphaned_else_start <= i <= orphaned_else_end:
            # Skip the orphaned else block entirely
            print(f"Removing orphaned else block line {i+1}: {line.rstrip()[:60]}")
            continue
        else:
            fixed_lines.append(line)
    
    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"\nFixed {end_line - start_line + 1} lines (un-indented by 4 spaces)")
    print(f"Removed {orphaned_else_end - orphaned_else_start + 1} lines (orphaned else block)")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    input_file = "/workspaces/pykt-toolkit/pykt/models/gainakt3_exp.py"
    output_file = "/workspaces/pykt-toolkit/pykt/models/gainakt3_exp.py"
    
    fix_indentation(input_file, output_file)
    print("\n✅ Indentation bug fixed!")
