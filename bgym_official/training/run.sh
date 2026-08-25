main_file='main.py' # [proteinMPNN & ESM2: main.py, OHE: main-OHE.py]
split='cluster' # [random, contig, mod, cluster]
use_weight='pretrained' # [pretrained, native], only for deep learning method
model_type='structure' # ['structure', 'sequence', 'OHE', 'OHE_AA']
mode='inter' # ['intra', 'inter']
batch_size=8

if [ $mode == 'intra' ]; then
  for i in `seq 0 24`
    do
        CUDA_VISIBLE_DEVICES=$1 python $main_file \
          --train_dms_mapping ../input/BindingGYM.csv \
          --dms_input ../input/Binding_substitutions_DMS \
          --dms_index $i --batch_size $batch_size \
          --split $split --use_weight $use_weight \
          --model_type $model_type --mode $mode --seed 42 
    done
else
    CUDA_VISIBLE_DEVICES=$1 python $main_file \
      --train_dms_mapping ../input/BindingGYM.csv \
      --dms_input ../input/Binding_substitutions_DMS \
      --batch_size $batch_size \
      --split $split --use_weight $use_weight \
      --model_type $model_type --mode $mode --seed 42 
fi