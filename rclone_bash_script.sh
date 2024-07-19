#!/bin/bash

# Run the rclone command and capture its output
rclone_path=/home/asravan2/rclone-linux-ppc64le.v1.63.0-beta.7033.fcf47a839

output=$($rclone_path lsd multiple_zebrafish_raw_dataset:overlapping_zebrafish_raw_videos/2D/free_swimming_2D_overlapping)

# Iterate over each line of the output
echo "$output" |while read -r line; do
        # Extract the last word of the line
        last_word=$(echo "$line" | awk '{print $NF}')
        echo "Creating folder: $last_word"
        mkdir $last_word
        echo "Copying: $last_word"
        $rclone_path copy multiple_zebrafish_raw_dataset:overlapping_zebrafish_raw_videos/2D/free_swimming_2D_overlapping/$last_word $last_word/
        echo "Deleting $last_word"
        rm -r $last_word
done

