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
