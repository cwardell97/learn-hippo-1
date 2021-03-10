#!/bin/bash

exp_name='Mental-Sims-v3.1_ms-p5_r5-use_V-F-real'
n_epoch=200
n_branch=4

def_prob=.25
n_def_tps=0

similarity_max=.4
similarity_min=0.0

p_rm_ob_enc=0.0   # effects noisey of observation (.3 it doesnt see all params)
p_rm_ob_rcl=0.0    # effects noisey of observation
pad_len=-1

eta=.1
lr=8e-4
sup_epoch=600

penalty_random=0
penalty_discrete=1 # never changed
penalty_onehot=0

normalize_return=1 # never changed

for subj_id in 1
do
    for penalty in 4
    do
        for n_param in 16
        do
            for enc_size in 16
            do
                for n_event_remember in 2
                do
                    for n_hidden in 194
                    do
                        for n_hidden_dec in 128
                        do
                            sbatch train-sl-ms.sh $exp_name \
                                ${subj_id} ${penalty} ${n_param} ${n_branch} \
                                ${n_hidden} ${n_hidden_dec} ${eta} ${lr} \
                                $n_epoch ${sup_epoch} \
                                ${p_rm_ob_enc} ${p_rm_ob_rcl} ${n_event_remember} \
                                ${pad_len} ${enc_size} \
                                ${similarity_max} ${similarity_min} \
                                ${penalty_random} ${penalty_discrete} ${penalty_onehot}\
                                ${normalize_return} ${def_prob} ${n_def_tps}
                        done
                    done
                done
            done
        done
    done
done
