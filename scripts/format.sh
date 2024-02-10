#!/bin/bash

# Run black on a single file
format_single_file() {
    file="$1"
    black "$file"
}

# Run black on all Python files in a directory recursively
format_all_files() {
    directory="$1"
    black "$directory"
}

# Check if the user provided a file or directory
if [ $# -eq 0 ]; then
    echo "Usage: $0 <file or directory>"
    exit 1
fi

# Check if the argument is a file or a directory
arg="$1"
if [ -f "$arg" ]; then
    format_single_file "$arg"
elif [ -d "$arg" ]; then
    format_all_files "$arg"
else
    echo "Error: '$arg' is not a valid file or directory"
    exit 1
fi
