#!/bin/bash

while read -p "Choose a resource to download [data/wiki/models/index]: " choice; do
    case "$choice" in
        data )
            TARGET=$choice
            TARGET_DIR=$DATA_DIR
            break ;;
        wiki )
            TARGET=$choice
            TARGET_DIR=$DATA_DIR
            break ;;
        models )
            TARGET=$choice
            TARGET_DIR=$SAVE_DIR
            break ;;
        index )
            TARGET=$choice
            TARGET_DIR=$SAVE_DIR
            break ;;
        * ) echo "Please type among [data/wiki/models/index]";
            exit 0 ;;
    esac
done

echo "$TARGET will be downloaded at $TARGET_DIR"

# Download + untar + rm
case "$TARGET" in
    data )
        wget -O "$TARGET_DIR/densephrases-data.tar.gz" "https://nlp.cs.princeton.edu/projects/densephrases/densephrases-data.tar.gz"
        tar -xzvf "$TARGET_DIR/densephrases-data.tar.gz" -C "$TARGET_DIR" --strip 1
        rm "$TARGET_DIR/densephrases-data.tar.gz" ;;
    wiki )
        wget -O "$TARGET_DIR/wikidump.tar.gz" "https://nlp.cs.princeton.edu/projects/densephrases/wikidump.tar.gz"
        tar -xzvf "$TARGET_DIR/wikidump.tar.gz" -C "$TARGET_DIR"
        rm "$TARGET_DIR/wikidump.tar.gz" ;;
    models )
        wget -O "$TARGET_DIR/outputs.tar.gz" "https://nlp.cs.princeton.edu/projects/densephrases/outputs.tar.gz"
        tar -xzvf "$TARGET_DIR/outputs.tar.gz" -C "$TARGET_DIR" --strip 1
        rm "$TARGET_DIR/outputs.tar.gz" ;;
    index )
        wget -O "$TARGET_DIR/densephrases-multi_wiki-20181220.tar.gz" "https://nlp.cs.princeton.edu/projects/densephrases/densephrases-multi_wiki-20181220.tar.gz"
        tar -xzvf "$TARGET_DIR/densephrases-multi_wiki-20181220.tar.gz" -C "$TARGET_DIR"
        rm "$TARGET_DIR/densephrases-multi_wiki-20181220.tar.gz" ;;
    * ) echo "Wrong target $TARGET";
        exit 0 ;;
esac

echo "Downloading $TARGET done!"
