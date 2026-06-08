cd "$(dirname "$0")"

nohup python3 main.py \
  --kg_dir ../LungKG/LungKG_fusion/output \
  --output_dir output \
  --max_communities 8000 \
  --num_questions 4 \
  --tail_ratio 0.75 \
  --concurrency 30 \
  > generate.log 2>&1 &
