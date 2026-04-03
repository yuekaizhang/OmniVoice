# === Equal-length benchmark (original) ===
# CUDA_VISIBLE_DEVICES=0 python3 -m omnivoice.cli.infer_batch --test_list rtf_evaluation/test_rtf.jsonl --warmup 5 --num_step 16 --batch_size 4 --res_dir results/rtf/
# CUDA_VISIBLE_DEVICES=0 python3 -m omnivoice.cli.infer_batch --test_list rtf_evaluation/test_rtf.jsonl --warmup 5 --num_step 16 --batch_size 4 --res_dir results/rtf_flash/ --attn_implementation flash_attention_2

# === Variable-length benchmark (5s~30s, no sorting → max padding variance) ===
# SDPA (default)
# CUDA_VISIBLE_DEVICES=0 python3 -m omnivoice.cli.infer_batch --test_list rtf_evaluation/varlen_rtf.jsonl --warmup 5 --num_step 16 --batch_size 4 --no_sort True --res_dir results/varlen_rtf/ # 848.44s   29.31s Average RTF: 0.0345

# FlashAttention-2
# CUDA_VISIBLE_DEVICES=0 python3 -m omnivoice.cli.infer_batch --test_list rtf_evaluation/varlen_rtf.jsonl --warmup 5 --num_step 16 --batch_size 4 --no_sort True --res_dir results/varlen_rtf_flash/ --attn_implementation flash_attention_2 # 845.78s   26.77s   0.0317

# === CUDA Graph Comparison (SDPA) ===

# Equal-length: SDPA baseline
# CUDA_VISIBLE_DEVICES=0 python3 -m omnivoice.cli.infer_batch --test_list rtf_evaluation/test_rtf.jsonl --warmup 5 --num_step 16 --batch_size 4 --res_dir results/rtf_sdpa/
# Equal-length: SDPA + CUDA Graph
# CUDA_VISIBLE_DEVICES=0 python3 -m omnivoice.cli.infer_batch --test_list rtf_evaluation/test_rtf.jsonl --warmup 5 --num_step 16 --batch_size 4 --res_dir results/rtf_sdpa_cg/ --use_cuda_graph True

# Variable-length: SDPA baseline
CUDA_VISIBLE_DEVICES=0 python3 -m omnivoice.cli.infer_batch --test_list rtf_evaluation/varlen_rtf.jsonl --warmup 5 --num_step 16 --batch_size 4 --no_sort True --res_dir results/varlen_rtf_sdpa/
# Variable-length: SDPA + CUDA Graph
CUDA_VISIBLE_DEVICES=0 python3 -m omnivoice.cli.infer_batch --test_list rtf_evaluation/varlen_rtf.jsonl --warmup 5 --num_step 16 --batch_size 4 --no_sort True --res_dir results/varlen_rtf_sdpa_cg/ --use_cuda_graph True
