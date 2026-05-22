#!/usr/bin/env python3
"""
Merge all JSONL files from paloma_validation directory into a single consolidated JSONL file.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

def merge_paloma_files(input_dir: str, output_file: str) -> None:
    """
    Merge all JSONL files from input_dir into a single output_file.
    
    Args:
        input_dir: Directory containing subdirectories with validation.jsonl files
        output_file: Path to the output merged JSONL file
    """
    
    input_path = Path(input_dir)
    unique_id = 0
    total_count = 0
    
    # Find all validation.jsonl files
    jsonl_files = sorted(input_path.glob("*/validation.jsonl"))
    
    print(f"Found {len(jsonl_files)} JSONL files to merge")
    
    with open(output_file, 'w') as out_f:
        for jsonl_file in jsonl_files:
            # Extract the split/source name from the parent directory
            split_name = jsonl_file.parent.name
            file_count = 0
            
            print(f"Processing: {split_name} ({jsonl_file})")
            
            try:
                with open(jsonl_file, 'r') as in_f:
                    for line in in_f:
                        if not line.strip():
                            continue
                        
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Skipping invalid JSON in {split_name}: {e}")
                            continue
                        
                        # Create merged entry
                        merged_entry: Dict[str, Any] = {
                            "id": unique_id,
                            "split": split_name,
                            "text": entry.get("text", ""),
                        }
                        
                        # Add all other metadata from the original entry
                        if isinstance(entry, dict):
                            for key, value in entry.items():
                                if key != "text":
                                    if key not in merged_entry:
                                        merged_entry[key] = value
                        
                        # Write to output
                        out_f.write(json.dumps(merged_entry) + '\n')
                        
                        unique_id += 1
                        file_count += 1
                        total_count += 1
                
                print(f"  ✓ Processed {file_count} entries from {split_name}")
                
            except Exception as e:
                print(f"  ✗ Error processing {jsonl_file}: {e}")
    
    print(f"\n✓ Successfully merged {total_count} total entries")
    print(f"✓ Output saved to: {output_file}")


if __name__ == "__main__":
    input_directory = "/data/user_data/nsrikant/data/paloma_validation"
    output_path = "/data/user_data/nsrikant/bbox_data/data/paloma_validation.jsonl"
    
    merge_paloma_files(input_directory, output_path)
