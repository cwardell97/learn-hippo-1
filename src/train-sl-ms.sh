#!/bin/bash
#SBATCH -t 17:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 4G

#SBATCH --job-name=lcarnn
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=end
#SBATCH --mail-user=cwardell@princeton.edu
#SBATCH --output /home/cwardell/slurm_log/lhms-%j.log

#module load anaconda

DATADIR=/tigress/cwardell/logs/learn-hippocampus/log

echo $(date)
#echo "exp_name: $1"
#echo "subj_id: $2"
#echo "penalty: $3"
#echo "n_param $4"
#echo "n_branch $5"
#echo "n_hidden $6"
#echo "n_hidden_dec $7"
#echo "eta $8"
#echo "lr $9"
#echo "n_epoch $10"
#echo "sup_epoch $11"
#echo "p_rm_ob_enc $12"
#echo "p_rm_ob_rcl $13"
#echo "n_event_remember $14"
#echo "pad_len $15"
#echo "enc_size $16"
#echo "similarity_max $17"
#echo "similarity_min $18"
#echo "penalty_random $19"
#echo "penalty_discrete $20"
#echo "penalty_onehot $21"
#echo "normalize_return $22"
#echo "def_prob $23"
#echo "n_def_tps $24"

srun python -u train-sl-ms.py --exp_name ${1} \
    --subj_id ${2} --penalty ${3} --n_param ${4} --n_branch ${5} \
    --n_hidden ${6} --n_hidden_dec ${7} --eta ${8} --lr ${9} \
    --n_epoch ${10} --sup_epoch ${11} \
    --p_rm_ob_enc ${12} --p_rm_ob_rcl ${13} --n_event_remember ${14} \
    --pad_len ${15} --enc_size ${16} \
    --similarity_max ${17} --similarity_min ${18} \
    --penalty_random ${19} --penalty_discrete ${20} --penalty_onehot ${21} \
    --normalize_return ${22} --def_prob ${23} --n_def_tps ${24} \
    --log_root $DATADIR

echo $(date)
