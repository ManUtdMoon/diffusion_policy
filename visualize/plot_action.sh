for algo in "resrl" "zprl"
do
python visualize/visual_explore_action.py \
  --action-key sum_action \
  --ratio 3 \
  --image data/outputs/plot_action/color.png \
  --input-dir data/outputs/plot_action/${algo}_1st_epi/ \
  --output data/outputs/plot_action/${algo}_epi.png
done