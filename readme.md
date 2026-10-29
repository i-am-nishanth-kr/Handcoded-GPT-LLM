# Download dataset from
* https://huggingface.co/datasets/itsme-nishanth/60k_stories/tree/main
# commands:
* python gpt2_hybrid.py test // just to sanity check to see if the code is in working condition
* python gpt2_hybrid.py train --data 20k_stories.parquet --max_steps 6200 --eval_interval 100 --eval_iters 10 --batch_size 8
* python gpt2_hybrid.py generate

Can run this in google colab for free
