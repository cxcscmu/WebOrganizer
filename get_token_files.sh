#!/bin/bash

DOCUMENTS_DIR="/data/user_data/gonilude/WebOrganizer/Corpus-30B-v2/documents"
TOKENS_DIR="/data/group_data/cx_group/WebOrganizer/Corpus-30B/tokens"
OUTPUT_DIR="/data/user_data/gonilude/WebOrganizer/Corpus-30B-v2/tokens"

# Check if directories exist
if [[ ! -d "$DOCUMENTS_DIR" ]]; then
    echo "Error: Documents directory '$DOCUMENTS_DIR' does not exist."
    exit 1
fi

if [[ ! -d "$TOKENS_DIR" ]]; then
    echo "Error: Tokens directory '$TOKENS_DIR' does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Initialize counters
matched_files=0
total_documents=0
missing_tokens=0

echo "Starting file matching and copying process..."
echo "Documents directory: $DOCUMENTS_DIR"
echo "Tokens directory: $TOKENS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Process each file in the documents directory
for doc_file in "$DOCUMENTS_DIR"/*; do
    
    # Get the base filename without path
    doc_basename=$(basename "$doc_file")
    total_documents=$((total_documents + 1))
    
    echo "Processing document: $doc_basename"
    
    # Extract the base name by removing known extensions
    # This handles patterns like: CC_shard_00000000_processed.jsonl.zst -> CC_shard_00000000_processed
    base_name="${doc_basename%.*}"
    
    found_token=false
    ext="npy"

    # Remove suffix patterns like "_skill" to match token file naming
    # This converts "CC_shard_00000000_processed_skill" -> "CC_shard_00000000_processed"
    if [[ "$base_name" =~ ^(.+)_skill$ ]]; then
        base_name="${BASH_REMATCH[1]}"
    fi
    
    token_file="$TOKENS_DIR/${base_name}.${ext}"
    
    if [[ -f "$token_file" ]]; then
        echo "  Found matching token file: $(basename "$token_file")"
        
        # Copy the token file to output directory
        if cp "$token_file" "$OUTPUT_DIR/"; then
            echo "  ✓ Successfully copied to: $OUTPUT_DIR/$(basename "$token_file")"
            matched_files=$((matched_files + 1))
            found_token=true
            # break
        else
            echo "  ✗ Error copying file"
        fi
    fi
    
    if [[ "$found_token" == false ]]; then
        echo "  ✗ No matching token file found for: $base_name"
        missing_tokens=$((missing_tokens + 1))
    fi
    
    echo ""
done

echo "=== Summary ==="
echo "Total document files processed: $total_documents"
echo "Successfully matched and copied: $matched_files"
echo "Missing token files: $missing_tokens"

if [[ $matched_files -eq 0 ]]; then
    echo ""
    echo "No files were copied. Please check:"
    echo "1. File naming patterns match between documents and tokens folders"
    echo "2. Token files have expected extensions (npy, pt, bin, tok, tokens)"
    echo "3. Directory paths are correct"
fi

echo ""
echo "Process completed. Check the output directory: $OUTPUT_DIR"