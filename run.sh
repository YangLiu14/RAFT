python demo.py --model /mnt/raid/davech2y/liuyang/model_weights/RAFT/raft-sintel.pth \
               --path /mnt/raid/davech2y/liuyang/data/MOTS20/train/



python gen_opt_flow.py --model /mnt/raid/davech2y/liuyang/model_weights/RAFT/raft-sintel.pth \
                       --path /mnt/raid/davech2y/liuyang/data/MOTS20/train/ \
                       --outdir /mnt/raid/davech2y/liuyang/Optical_Flow/MOTS20_RAFT_sintel/



python gen_opt_flow.py --model /mnt/raid/davech2y/liuyang/model_weights/RAFT/raft-sintel.pth \
                       --path /mnt/raid/davech2y/liuyang/data/TAO/frames/val/ \
                       --outdir /mnt/raid/davech2y/liuyang/Optical_Flow/TaoVal_RAFT_sintel/ \
                       --datasrc Charades

python gen_opt_flow.py --model /storage/slurm/liuyang/model_weights/RAFT/raft-sintel.pth \
                       --path /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                       --outdir /storage/slurm/liuyang/Optical_Flow/TaoVal_RAFT_sintel/ \
                       --datasrc ArgoVerse