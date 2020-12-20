BASE_DAVE=/mnt/raid/davech2y/liuyang/
BASE_TUM=/storage/slurm/liuyang/

python demo.py --model /mnt/raid/davech2y/liuyang/model_weights/RAFT/raft-sintel.pth \
               --path /mnt/raid/davech2y/liuyang/data/MOTS20/train/



python gen_opt_flow.py --model /mnt/raid/davech2y/liuyang/model_weights/RAFT/raft-sintel.pth \
                       --path /mnt/raid/davech2y/liuyang/data/MOTS20/train/ \
                       --outdir /mnt/raid/davech2y/liuyang/Optical_Flow/MOTS20_RAFT_sintel/



python gen_opt_flow.py --model /mnt/raid/davech2y/liuyang/model_weights/RAFT/raft-sintel.pth \
                       --path /mnt/raid/davech2y/liuyang/data/TAO/frames/val/ \
                       --outdir /mnt/raid/davech2y/liuyang/Optical_Flow/TaoVal_RAFT_sintel_downscaled/ \
                       --datasrc YFCC100M

python gen_opt_flow.py --model /storage/slurm/liuyang/model_weights/RAFT/raft-sintel.pth \
                       --path /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                       --outdir /storage/slurm/liuyang/Optical_Flow/TaoVal_RAFT_sintel_downscaled/ \
                       --datasrc ArgoVerse

python opt_flow_warp.py --prop_dir /mnt/raid/davech2y/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/boxNMS/_objectness/ \
                        --opt_flow_dir /mnt/raid/davech2y/liuyang/Optical_Flow/TaoVal_RAFT_sintel/ \
                        --out_dir /mnt/raid/davech2y/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/opt_flow_output/_objectness \
                        --datasrc ArgoVerse

python opt_flow_warp_subset.py --prop_dir /mnt/raid/davech2y/liuyang/TAO_eval/TAO_VAL_Proposals/subset_experiment/Panoptic_Cas_R101_NMSoff_forTracking_Embed/boxNMS/_objectness/ \
                        --opt_flow_dir /mnt/raid/davech2y/liuyang/Optical_Flow/TaoVal_RAFT_sintel/ \
                        --out_dir /mnt/raid/davech2y/liuyang/TAO_eval/TAO_VAL_Proposals/subset_experiment/Panoptic_Cas_R101_NMSoff_forTracking_Embed/opt_flow_output/_objectness \
                        --datasrc YFCC100M



python opt_flow_warp_subset.py --prop_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/subset_experiment/Panoptic_Cas_R101_NMSoff_forTracking_Embed/boxNMS/_objectness/ \
                        --opt_flow_dir /storage/slurm/liuyang/Optical_Flow/TaoVal_RAFT_sintel/ \
                        --out_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/subset_experiment/Panoptic_Cas_R101_NMSoff_forTracking_Embed/opt_flow_output/_objectness \
                        --datasrc ArgoVerse

python opt_flow_warp.py --prop_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/boxNMS_npz002/_objectness/ \
                        --opt_flow_dir /storage/slurm/liuyang/Optical_Flow/TaoVal_RAFT_sintel_downscaled/ \
                        --out_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/opt_flow_output/_objectness \
                        --datasrc ArgoVerse