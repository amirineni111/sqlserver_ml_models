#!/usr/bin/env python3
"""
Backup Cleanup Utility

Cleans up old backup files from data/backups directory.
Keeps only the N most recent backup sets to save disk space.

Usage:
    python cleanup_old_backups.py              # Keep last 5 backup sets (default)
    python cleanup_old_backups.py --keep 3     # Keep last 3 backup sets
    python cleanup_old_backups.py --list-only  # List files without deleting
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def list_backup_files(backup_dir):
    """List all backup files grouped by timestamp"""
    backup_sets = defaultdict(list)
    
    for file in backup_dir.glob('*'):
        if file.is_file():
            # Extract timestamp from filename (format: YYYYMMDD_HHMMSS_filename)
            parts = file.name.split('_')
            if len(parts) >= 3:
                timestamp = f"{parts[0]}_{parts[1]}"
                backup_sets[timestamp].append(file)
    
    return backup_sets

def cleanup_old_backups(keep_count=5, list_only=False):
    """Clean up old backup files, keeping only the most recent sets"""
    
    backup_dir = Path('data/backups')
    
    if not backup_dir.exists():
        print(f"[INFO] Backup directory does not exist: {backup_dir}")
        return
    
    print("=" * 70)
    print("BACKUP CLEANUP UTILITY")
    print("=" * 70)
    print()
    
    # List backup sets
    backup_sets = list_backup_files(backup_dir)
    
    if not backup_sets:
        print("[INFO] No backup files found")
        return
    
    # Sort by timestamp (newest first)
    sorted_timestamps = sorted(backup_sets.keys(), reverse=True)
    
    print(f"📁 Found {len(sorted_timestamps)} backup sets:")
    print()
    
    total_size = 0
    for idx, timestamp in enumerate(sorted_timestamps, 1):
        files = backup_sets[timestamp]
        set_size = sum(f.stat().st_size for f in files)
        total_size += set_size
        
        status = "KEEP" if idx <= keep_count else "DELETE"
        status_icon = "✅" if idx <= keep_count else "🗑️"
        
        print(f"{status_icon} {status:<8} {timestamp}  ({len(files)} files, {set_size/1024/1024:.1f} MB)")
        for file in files:
            action = "   " if idx <= keep_count else "   - "
            print(f"      {action}{file.name}")
    
    print()
    print(f"📊 Total backup size: {total_size/1024/1024:.1f} MB")
    print()
    
    # Determine what to delete
    sets_to_delete = sorted_timestamps[keep_count:]
    
    if not sets_to_delete:
        print(f"[INFO] All backups are within keep limit ({keep_count}). Nothing to delete.")
        return
    
    files_to_delete = []
    for timestamp in sets_to_delete:
        files_to_delete.extend(backup_sets[timestamp])
    
    deleted_size = sum(f.stat().st_size for f in files_to_delete)
    
    print(f"🗑️  Will delete {len(files_to_delete)} files from {len(sets_to_delete)} backup sets")
    print(f"💾 Will free up {deleted_size/1024/1024:.1f} MB of disk space")
    print()
    
    if list_only:
        print("[INFO] List-only mode - no files deleted")
        return
    
    # Confirm deletion
    response = input("Proceed with deletion? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("[INFO] Cleanup cancelled")
        return
    
    # Delete files
    deleted_count = 0
    for file in files_to_delete:
        try:
            file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to delete {file.name}: {e}")
    
    print()
    print(f"✅ Cleanup complete!")
    print(f"   Deleted: {deleted_count} files")
    print(f"   Freed: {deleted_size/1024/1024:.1f} MB")
    print(f"   Kept: {len(sorted_timestamps) - len(sets_to_delete)} most recent backup sets")
    print()

def main():
    parser = argparse.ArgumentParser(description='Clean up old backup files')
    parser.add_argument('--keep', type=int, default=5,
                       help='Number of most recent backup sets to keep (default: 5)')
    parser.add_argument('--list-only', action='store_true',
                       help='List files that would be deleted without actually deleting')
    
    args = parser.parse_args()
    
    cleanup_old_backups(keep_count=args.keep, list_only=args.list_only)

if __name__ == "__main__":
    main()
