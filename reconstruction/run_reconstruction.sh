#!/usr/bin/env bash

# Uncomment one command at a time.

# python reconstruction/run/recon_sam3d.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --num-workers 1
# python reconstruction/run/recon_sam3d_tsdf.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --num-workers 4
# python reconstruction/run/recon_sam3d_dmesh.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1
# python reconstruction/run/recon_sam3d_tsdf_dmesh.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1

# python reconstruction/run/recon_hunyuan3d.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --num-workers 1
# python reconstruction/run/recon_hunyuan3d_tsdf.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --num-workers 4
# python reconstruction/run/recon_hunyuan3d_dmesh.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1
# python reconstruction/run/recon_hunyuan3d_tsdf_dmesh.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1

# python reconstruction/reconstruct_instantmesh.py --help
# python reconstruction/reconstruct.py --help
# python reconstruction/point_reconstruct.py --help
