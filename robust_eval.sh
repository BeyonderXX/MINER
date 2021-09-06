# 测试 baseline
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/baseline --do_train --do_eval --do_predict --baseline --do_robustness_eval > bert_bsl.log 2>&1 &

nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/baseline --do_predict --baseline --do_robustness_eval > bert_bsl.log 2>&1 &


# 测试无 regular z
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/wo_reg_z --do_train --do_eval --do_predict --regular_z --do_robustness_eval > bert_wo_reg_z.log 2>&1 &

nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/wo_reg_z --do_predict --regular_z --do_robustness_eval > bert_wo_reg_z.log 2>&1 &

# 测试 beta 5e-4
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_4 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-4 > bert_beta_5e_4.log 2>&1 &
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_4 --do_predict --do_robustness_eval --beta 5e-4 > bert_beta_5e_4.log 2>&1 &


nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_5 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bert_beta_5e_5.log 2>&1 &
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_5 --do_predict --do_robustness_eval --beta 5e-5 > bert_beta_5e_5.log 2>&1 &


nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6.log 2>&1 &
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6 --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6.log 2>&1 &

# 增大 beta 实验
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/beta_1e_3 --do_train --do_eval --do_predict --do_robustness_eval --beta 1e-3 > bert_beta_1e_3.log 2>&1 &

nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_3 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-3 > bert_beta_5e_3.log 2>&1 &
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/beta_1e_1 --do_train --do_eval --do_predict --do_robustness_eval --beta 1e-1 > bert_beta_1e_1.log 2>&1 &
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/beta_1e_2 --do_train --do_eval --do_predict --do_robustness_eval --beta 1e-2 > bert_beta_1e_2.log 2>&1 &

# 精调 beta
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/beta_1e_5 --do_train --do_eval --do_predict --do_robustness_eval --beta 1e-5 > bert_beta_1e_5.log 2>&1 &
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/beta_1e_6 --do_train --do_eval --do_predict --do_robustness_eval --beta 1e-6 > bert_beta_1e_6.log 2>&1 &
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_7 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-7 > bert_beta_5e_7.log 2>&1 &


--gpu_id 4 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_5 --do_predict --beta 5e-5
--gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/wo_reg_z --do_predict --regular_z


# hidden dim 实验
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h200 --hidden_dim 200 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h200.log 2>&1 &
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h100 --hidden_dim 100 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h100.log 2>&1 &
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h50 --hidden_dim 50 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h50.log 2>&1 &

nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h500 --hidden_dim 500 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h500.log 2>&1 &
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h756 --hidden_dim 756 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h756.log 2>&1 &


# 第一版entity regularizer实验（gamma 1e-03  total_typos）
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3/ --batch_size 60 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bert_beta_5e_5_h300_ent_typos.log 2>&1 &
