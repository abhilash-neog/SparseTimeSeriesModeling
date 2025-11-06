bash scripts/EXP-LongForecasting/LSTM/etth2_masked_Spline.sh "p6 p7" 0 1 mcar 1
bash scripts/EXP-LongForecasting/LSTM/etth2_masked_Spline.sh "p6 p7" 0 2 mcar 1

bash scripts/EXP-LongForecasting/LSTM/etth2_masked_SAITS.sh "p6 p7" 1 1 mcar 1
bash scripts/EXP-LongForecasting/LSTM/etth2_masked_SAITS.sh "p6 p7" 1 2 mcar 1

bash scripts/EXP-LongForecasting/LSTM/etth2_masked.sh "p6 p7" 4 1 mcar h2_mtsm_mcar_trial1 1 tfi 1 1 1 1 1
bash scripts/EXP-LongForecasting/LSTM/etth2_masked.sh "p6 p7" 4 2 mcar h2_mtsm_mcar_trial2 1 tfi 1 1 1 1 1
