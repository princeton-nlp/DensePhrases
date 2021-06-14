#!/bin/bash

while read -p "Choose a resource to download [data/models/index]: " choice; do
    case "$choice" in
        data )
            TARGET=$choice
            TARGET_DIR=$DPH_DATA_DIR
            break ;;
        models )
            TARGET=$choice
            TARGET_DIR=$DPH_SAVE_DIR
            break ;;
        index )
            TARGET=$choice
            TARGET_DIR=$DPH_SAVE_DIR
            break ;;
        * ) echo "Please type among [data/models/index]";
            exit 0 ;;
    esac
done

echo "$TARGET will be downloaded at $TARGET_DIR"

# Download + untar + rm
case "$TARGET" in
    data )
        wget -O "$TARGET_DIR/dph-data.tar.gz" "https://nlp.cs.princeton.edu/projects/densephrases/dph-data.tar.gz"
        tar -xzvf "$TARGET_DIR/dph-data.tar.gz" -C "$TARGET_DIR" --strip 1
        rm "$TARGET_DIR/dph-data.tar.gz" ;;
    models )
        wget -O "$TARGET_DIR/outputs.tar.gz" "https://nlp.cs.princeton.edu/projects/densephrases/outputs.tar.gz"
        tar -xzvf "$TARGET_DIR/outputs.tar.gz" -C "$TARGET_DIR" --strip 1
        rm "$TARGET_DIR/outputs.tar.gz" ;;
    index )
        wget -O "$TARGET_DIR/dph-nqsqd-pb2_20181220_concat.tar.gz" "https://nlp.cs.princeton.edu/projects/densephrases/dph-nqsqd3-multi5-pb2_1_20181220_concat.tar.gz"
        tar -xzvf "$TARGET_DIR/dph-nqsqd-pb2_20181220_concat.tar.gz" -C "$TARGET_DIR"
        rm "$TARGET_DIR/dph-nqsqd-pb2_20181220_concat.tar.gz" ;;
    * ) echo "Wrong target $TARGET";
        exit 0 ;;
esac

echo "Downloading $TARGET done!"
