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


# 不注释 z_reg, MSE, 错误结果保存至 /root/RobustNER/out/bn_oov_1_mi_0_bad, bn_oov_1_mi_0.log
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bn_oov_1_mi_0/ --gama 1e-2 --r 0 --epoch 40 --rep_mode typos --batch_size 24 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_0_test.log 2>&1 &
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bn_oov_1_mi_3/ --gama 1e-2 --r 1e-3 --epoch 40 --rep_mode typos --batch_size 28 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_3.log 2>&1 &

# 注释 z_reg, MSE
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bn_oov_1_mi_e4_mse/ --gama 1e-2 --r 1e-4 --epoch 40 --rep_mode typos --batch_size 16 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e4_mse.log 2>&1 &

# 注释 z_reg, 0 info params
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_oov_1_mi_0_wo_z_params/ --gama 1e-2 --r 0 --epoch 40 --rep_mode typos --batch_size 16 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_0_wo_z_params.log 2>&1 &
# 注释 z_reg, 1e-4 info params
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bn_oov_1_mi_e4_wo_z_params/ --gama 1e-2 --r 1e-4 --epoch 40 --rep_mode typos --batch_size 16 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e4_wo_z_params.log 2>&1 &

# softplus -> leaklyrelu  + 0 params
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bn_oov_1_mi_0_lrelu/ --gama 1e-2 --r 0 --epoch 40 --rep_mode typos --batch_size 16 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_0_lrelu.log 2>&1 &
# softplus -> leaklyrelu + 1e-4 params
nohup python main.py --gpu_id 5 --output_dir /root/RobustNER/out/bn_oov_1_mi_e4_lrelu/ --gama 1e-2 --r 1e-4 --epoch 40 --rep_mode typos --batch_size 16 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e4_lrelu.log 2>&1 &
# relu -> Tanh + 1e-4 params
nohup python main.py --gpu_id 6 --output_dir /root/RobustNER/out/bn_oov_1_mi_e4_tanh/ --gama 1e-2 --r 1e-4 --epoch 40 --rep_mode typos --batch_size 16 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e4_tanh.log 2>&1 &
# relu -> Tanh + softplus -> leaklyrelu + 1e-4 params
nohup python main.py --gpu_id 7 --output_dir /root/RobustNER/out/bn_oov_1_mi_e4_tanh_leaklyrelu/ --gama 1e-2 --r 1e-4 --epoch 40 --rep_mode typos --batch_size 16 --do_train --do_eval --do_predict --do_robustness_eval > bn_oov_1_mi_e4_tanh_leaklyrelu.log 2>&1 &


# 17 sample 机制
