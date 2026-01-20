for f in /home/nsrikant/BehaviorBoxNew/results/blimp_full/*/blimp_summary_seq_logprob.json; do
    mv "$f" "$(dirname "$f")/blimp_full_summary_seq_logprob.json"
done