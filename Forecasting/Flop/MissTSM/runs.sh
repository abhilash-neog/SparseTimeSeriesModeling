
sbatch -J m2mstsm6789per3 --time 08:30:00 --partition dgx_normal_q ./scripts/ARC_scripts/ETTm2_masking_ARC.sh "a6 a7 a8 a9" 3 periodic "96 192 336 720"
sbatch -J m2mstsm6789per4 --time 08:30:00 --partition dgx_normal_q ./scripts/ARC_scripts/ETTm2_masking_ARC.sh "a6 a7 a8 a9" 4 periodic "96 192 336 720"
sbatch -J h2mstsm6789per0 --time 05:30:00 --partition dgx_normal_q ./scripts/ARC_scripts/ETTh2_masking_ARC.sh "a6 a7 a8 a9" 0 periodic "96 192 336 720"
sbatch -J h2mstsm6789per1 --time 05:30:00 --partition dgx_normal_q ./scripts/ARC_scripts/ETTh2_masking_ARC.sh "a6 a7 a8 a9" 1 periodic "96 192 336 720"
sbatch -J h2mstsm6789per2 --time 05:30:00 --partition dgx_normal_q ./scripts/ARC_scripts/ETTh2_masking_ARC.sh "a6 a7 a8 a9" 2 periodic "96 192 336 720"
sbatch -J h2mstsm6789per3 --time 05:30:00 --partition dgx_normal_q ./scripts/ARC_scripts/ETTh2_masking_ARC.sh "a6 a7 a8 a9" 3 periodic "96 192 336 720"
sbatch -J h2mstsm6789per4 --time 05:30:00 --partition dgx_normal_q ./scripts/ARC_scripts/ETTh2_masking_ARC.sh "a6 a7 a8 a9" 4 periodic "96 192 336 720"