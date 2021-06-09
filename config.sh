#!/bin/bash

# Modify below to your choice of directory
export BASE_DIR=./

while read -p "Use to $BASE_DIR as the base directory (requires at least 220GB for the installation)? [yes/no]: " choice; do
    case "$choice" in
        yes )
            break ;;
        no )
            while read -p "Type in the directory: " choice; do
                case "$choice" in
                    * )
                        export BASE_DIR=$choice;
                        echo "Base directory set to $BASE_DIR";
                        break ;;
                esac
            done
            break ;;
        * ) echo "Please answer yes or no.";
            exit 0 ;;
    esac
done

# DPH_DATA_DIR: for datasets (including 'kilt', 'open-qa', 'single-qa', 'truecase', 'wikidump')
# DPH_SAVE_DIR: for pre-trained models or dumps; new models and dumps will also be saved here
# DPH_CACHE_DIR: for cache files from huggingface transformers
export DPH_DATA_DIR=$BASE_DIR/dph-data
export DPH_SAVE_DIR=$BASE_DIR/outputs
export DPH_CACHE_DIR=$BASE_DIR/cache

# Create directories
mkdir -p $DPH_DATA_DIR
mkdir -p $DPH_SAVE_DIR
mkdir -p $DPH_SAVE_DIR/logs
mkdir -p $DPH_CACHE_DIR

printf "\nEnvironment variables are set as follows:\n"
echo "DPH_DATA_DIR=$DPH_DATA_DIR"
echo "DPH_SAVE_DIR=$DPH_SAVE_DIR"
echo "DPH_CACHE_DIR=$DPH_CACHE_DIR"

# Append to bashrc, instructions
while read -p "Add to ~/.bashrc (recommended)? [yes/no]: " choice; do
    case "$choice" in
        yes )
            echo -e "\n# DensePhrases setup" >> ~/.bashrc;
            echo "export DPH_DATA_DIR=$DPH_DATA_DIR" >> ~/.bashrc;
            echo "export DPH_SAVE_DIR=$DPH_SAVE_DIR" >> ~/.bashrc;
            echo "export DPH_CACHE_DIR=$DPH_CACHE_DIR" >> ~/.bashrc;
            break ;;
        no )
            break ;;
        * ) echo "Please answer yes or no." ;;
    esac
done
