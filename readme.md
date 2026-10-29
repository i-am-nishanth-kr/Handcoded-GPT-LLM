# download dataset from
* https://huggingface.co/datasets/itsme-nishanth/60k_stories/tree/main
# commands:
* python gpt2_hybrid.py train --data 60k_stories.parquet --max_steps 200 --eval_interval 100 --eval_iters 10 --batch_size 8
* python gpt2_hybrid.py generate
