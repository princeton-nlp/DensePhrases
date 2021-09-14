#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=paq-dump

# Remove one # to uncommment
#SBATCH --output=%x-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=0-11:00:00    # Run for 6 hours
#SBATCH --gres=gpu:1

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=NONE
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=jl5167@cs.princeton.edu


# Define and create a unique scratch directory for this job
tag=paq-dump-wiki;

# Submit jobs.
version=4
# for model in 'densephrases-multi'; do
# for model in 'densephrases-sbs-psg-3e-5-ddp4/checkpoint-300000'; do
for model in 'densephrases-paq-300k-multi5'; do
  # for st in 0,400 400,800 800,1200 1200,1600 1600,2000 2000,2400 2400,2800 2800,3233; do 
  for st in 2800,3233; do 
  # for st in 0,2 2,4 4,6 6,8 8,10 10,12 12,14 14,16; do 
    # You can use srun to run multiple scripts in the same job in parallel (make sure to use & at the end!). Note how you can specify the resources used for each srun and make them exclusive using the --exclusive flag.
    IFS=","; set -- $st; echo $1:$2;
    srun --gres=gpu:1 -n 1 --mem=24G --exclusive python generate_phrase_vecs.py --model_type bert --pretrained_name_or_path SpanBERT/spanbert-base-cased --cache_dir ${CACHE_DIR} --data_dir ${DATA_DIR}/wikidump/wiki-dpr-rand --predict_file $1:$2 --do_dump --max_seq_length 200 --doc_stride 180 --fp16 --load_dir ${SAVE_DIR}/${model} --output_dir ${SAVE_DIR}/densephrases-paq-300k-multi5_wiki-dpr-rand --filter_threshold 0.0 --append_title &
  done;
done;
wait; #Make sure to wait till all the runs have completed.

# Finish the script
exit 0
