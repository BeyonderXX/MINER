nohup python -u main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_span_test/ --epoch 30  --baseline --regular_z --regular_norm --regular_entity --regular_context --batch_size 128 --do_train --do_eval --do_predict --do_robustness_eval > bert_span_test.log 2>&1 &


nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_50_oov_1/ --gama 1 --epoch 50 --rep_mode typos --batch_size 64 --do_train --do_eval --do_predict --do_robustness_eval > bn_50_oov_1.log 2>&1 &

nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_50_oov_1/ --gama 100 --epoch 50 --rep_mode typos --batch_size 64 --do_train --do_eval --do_predict --do_robustness_eval > bn_50_oov_1.log 2>&1 &

# 25 服务器
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_50_oov_1/ --classifier_lr 1e-3 --gama 100 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_50_oov_1.log 2>&1 &
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bn_50_oov_1_5e4/ --classifier_lr 5e-4 --gama 100 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_50_oov_1_5e4.log 2>&1 &

nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bn_oov_1_mi_e3/ --gama 1 --r 1e-3 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e3.log 2>&1 &
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bn_oov_10_mi_e3/ --gama 10 --r 1e-3 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_10_mi_e3.log 2>&1 &
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_oov_1_mi_e4/ --gama 1 --r 1e-4 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e4.log 2>&1 &
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bn_oov_10_mi_e4/ --gama 10 --r 1e-4 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_10_mi_e4.log 2>&1 &

nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bn_oov_1_mi_e2/ --gama 1 --r 1e-2 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e2.log 2>&1 &
nohup python main.py --gpu_id 5 --output_dir /root/RobustNER/out/bn_oov_10_mi_e2/ --gama 10 --r 1e-2 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_10_mi_e2.log 2>&1 &


nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bn_oov_1_mi_e3_r_e1/ --gama 1 --r 1e-1 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e3_r_e1.log 2>&1 &


nohup python -u main.py --gpu_id 4 --output_dir /root/RobustNER/out/bn_oov_1_mi_e2/ --gama 1 --r 1e-2 --epoch 1 --rep_mode typos --batch_size 32 --do_train > test.log 2>&1 &


# MI debug, remove MI loss
nohup python -u main.py --gpu_id 0 --output_dir /root/RobustNER/out/bn_oov_1_mi_test/ --gama 1 --epoch 50 --rep_mode typos --batch_size 32 --do_train > bn_oov_1_mi_test.log 2>&1 &

# remove oov loss
nohup python -u main.py --gpu_id 1 --output_dir /root/RobustNER/out/bn_mi_e2/ --r 1e-2 --epoch 50 --rep_mode typos --batch_size 32 --do_train > bn_mi_e2.log 2>&1 &
# remove MI + oov loss
nohup python -u main.py --gpu_id 1 --output_dir /root/RobustNER/out/bn_test/ --gama 0 --r 0 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_test.log 2>&1 &


# revert test
nohup python main.py --gpu_id 0 --output_dir /root/RobustNERCom_revert/out/bn_oov_1_mi_0/ --gama 1 --r 0 --epoch 40 --rep_mode typos --batch_size 64 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_0.log 2>&1 &
nohup python main.py --gpu_id 4 --output_dir /root/RobustNERCom_revert/out/bn_oov_1_mi_e4/ --gama 1 --r 1e-4 --epoch 40 --rep_mode typos --batch_size 64 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e4.log 2>&1 &


# 修改项： 1.参数注释； 2.InfoNCE实现； 3.MSE->KL散度； 4.类接口; 5.sample size
# 无修改版本
# CrossCategory 58.03    Typos 86.13   OOV 74.38
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/origin_test/ --gama 1 --r 1e-2 --epoch 30 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > origin_test.log 2>&1 &

# InfoNCE取消注释
# CrossCategory 45.22    Typos 74.37   OOV 65.41
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/nce_opt_test/ --gama 1 --r 1e-2 --epoch 30 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > nce_opt_test.log 2>&1 &

# InfoNCE实现修改
# CrossCategory 56.17    Typos 84.66   OOV 72.24
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/nce_new_test/ --gama 1 --r 1e-2 --epoch 30 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > nce_new_test.log 2>&1 &

# MSE -> KL散度
# CrossCategory 56.0    Typos 84.33  OOV 73.17
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/kl_test/ --gama 1e-3 --r 1e-2 --epoch 30 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > kl_test.log 2>&1 &

# sample size
# CrossCategory 0.49    Typos 0.37   OOV 0.60
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/sample_5_test/ --gama 1 --r 1e-2 --epoch 30 --rep_mode typos --batch_size 28 --do_train --do_eval --do_predict --do_robustness_eval > sample_5_test.log 2>&1 &

# 类接口全改
# CrossCategory 3.81    Typos 7.51    OOV 4.39
nohup python main.py --gpu_id 5 --output_dir /root/RobustNER/out/inter_new/ --gama 1e-3 --r 1e-2 --epoch 30 --rep_mode typos --batch_size 28 --do_train --do_eval --do_predict --do_robustness_eval > inter_new.log 2>&1 &

# 类接口全改， sample_size 1
# CrossCategory 53.95   Typos 85.04   OOV 71.61
nohup python main.py --gpu_id 6 --output_dir /root/RobustNER/out/inter_new_sample1/ --gama 1e-3 --r 1e-2 --epoch 30 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > inter_new_sample1.log 2>&1 &


# 新实现 + lr 5e-5
# origin 88.98  CrossCategory 59.19   Typos 86.48   OOV 75.78
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/best_params_test/ --gama 1 --r 1e-3 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > best_params_test.log 2>&1 &

# lr 测试 5e-4
# origin 30.77  CrossCategory 12.17   Typos 33.16   OOV 22.63
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/best_params_5e4/ --gama 1 --bert_lr 5e-4 --lr 5e-4 --r 1e-3 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > best_params_5e4.log 2>&1 &

# lr 测试 1e-4
# origin 87.91  CrossCategory 56.41   Typos 83.87   OOV 71.71
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/best_params_1e4/ --gama 1 --bert_lr 1e-4 --lr 1e-4 --r 1e-3 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > best_params_1e4.log 2>&1 &

# lr 测试 1e-5
# origin 89.80  CrossCategory 61.39   Typos 87.34   OOV 77.29
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/best_params_1e5/ --gama 1 --bert_lr 1e-5 --lr 1e-5 --r 1e-3 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > best_params_1e5.log 2>&1 &

# bert 1e-5, lr 5e-5
# origin 90.41  CrossCategory 60.99   Typos 87.33   OOV 77.77
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/best_params_b1e5_l5e5/ --gama 1 --bert_lr 1e-5 --lr 5e-5 --r 5e-2 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > best_params_b1e5_l5e5.log 2>&1 &

# 解除 w 注释
# origin 88.63  CrossCategory 59.93   Typos 87.12   OOV 75.25
nohup python main.py --gpu_id 5 --output_dir /root/RobustNER/out/best_params_infoNCE/ --gama 1 --bert_lr 5e-5 --lr 5e-5 --r 5e-3 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > best_params_infoNCE.log 2>&1 &



# ***************************** CONLL2003 消融实验 ***********************************
# bsl + MI
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/ablation_mi/ --trans_weight 0 --gama 0 --r 5e-2 --bert_lr 1e-5 --lr 5e-5  --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > ablation_mi.log 2>&1 &
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/ablation_mi_trans/ --trans_weight 1 --gama 0 --r 5e-2 --bert_lr 1e-5 --lr 5e-5  --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > ablation_mi_trans.log 2>&1 &


# bsl + OOV
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/ablation_oov/ --trans_weight 0 --gama 1 --r 0 --bert_lr 1e-5 --lr 5e-5  --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > ablation_oov.log 2>&1 &
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/ablation_oov_trans/ --trans_weight 1 --gama 1 --r 0 --bert_lr 1e-5 --lr 5e-5  --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > ablation_oov_trans.log 2>&1 &


# ***************************** OpenNER 实验 ***********************************
nohup python main_openNER.py --gpu_id 4 --output_dir /root/RobustNER/out/OpenNER/best_params_test/ --bert_lr 1e-5 --lr 5e-5 --gama 1 --r 1e-3 --epoch 50 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict > openNER_best_params_test.log 2>&1 &

