#!/usr/bin/env python
"""
Quick test to verify that running notes are saved correctly
"""

import os
import json

output_dir = "run_principled_test"
notes_file = os.path.join(output_dir, "running_notes.json")

print("=" * 60)
print("Running Notes Check")
print("=" * 60)

# Check if directory exists
if os.path.exists(output_dir):
    print(f"✅ Output directory exists: {output_dir}")

    # List all files
    files = []
    for root, dirs, filenames in os.walk(output_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, output_dir)
            files.append(rel_path)

    if files:
        print(f"\n📁 Files in {output_dir}:")
        for f in files:
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"   - {f} ({size:,} bytes)")
    else:
        print(f"\n⚠️  No files found in {output_dir}")

    # Check for notes file
    if os.path.exists(notes_file):
        print(f"\n✅ Running notes file found!")
        print(f"   Path: {notes_file}")

        # Read and display summary
        with open(notes_file, "r") as f:
            notes = json.load(f)

        print(f"\n📊 Summary:")
        print(f"   Total messages: {notes.get('total_messages', 0)}")
        print(f"   Stop reason: {notes.get('stop_reason', 'N/A')}")

        # Count message types
        msg_types = {}
        for msg in notes.get('messages', []):
            msg_type = msg.get('type', 'Unknown')
            msg_types[msg_type] = msg_types.get(msg_type, 0) + 1

        print(f"\n📝 Message types:")
        for msg_type, count in sorted(msg_types.items()):
            print(f"   - {msg_type}: {count}")

    else:
        print(f"\n❌ Running notes file NOT found: {notes_file}")
        print("\nThis file should be created after the inference completes.")

else:
    print(f"❌ Output directory does not exist: {output_dir}")

print("=" * 60)
