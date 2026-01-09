#!/bin/bash
# Real-time accuracy monitoring for federated learning
# Usage: bash scripts/monitor_accuracy_realtime.sh

WORKSPACE=${WORKSPACE:-outputs/nvflare_workspace}

echo "=== Real-time Accuracy Monitor ==="
echo "Monitoring accuracy from all sites..."
echo "Press Ctrl+C to stop"
echo ""

# Function to format and display accuracy from a log line
format_accuracy() {
    local line="$1"
    local site="$2"
    
    # Extract SUMMARY line
    if echo "$line" | grep -q "SUMMARY:"; then
        echo "$line" | sed "s/SUMMARY: /[Site $site] /"
    # Extract Round Complete
    elif echo "$line" | grep -q "Round.*Complete"; then
        echo "$line" | sed "s/Round /[Site $site] Round /"
    # Extract Train/Val/Test Accuracy
    elif echo "$line" | grep -qE "Train Accuracy|Val Accuracy|Test Accuracy"; then
        echo "$line" | sed "s/  /[Site $site] /"
    fi
}

# Monitor all site logs in real-time
tail -f ${WORKSPACE}/site-*/log.txt 2>/dev/null | while read line; do
    # Determine which site this log line is from
    if echo "$line" | grep -q "site-1"; then
        format_accuracy "$line" "1"
    elif echo "$line" | grep -q "site-2"; then
        format_accuracy "$line" "2"
    elif echo "$line" | grep -q "site-3"; then
        format_accuracy "$line" "3"
    # Check for accuracy-related lines
    elif echo "$line" | grep -qE "SUMMARY:|Round.*Complete|Train Accuracy|Val Accuracy|Test Accuracy"; then
        # Try to extract site from path or content
        if echo "$line" | grep -q "Site 1"; then
            format_accuracy "$line" "1"
        elif echo "$line" | grep -q "Site 2"; then
            format_accuracy "$line" "2"
        elif echo "$line" | grep -q "Site 3"; then
            format_accuracy "$line" "3"
        else
            echo "$line"
        fi
    fi
done


