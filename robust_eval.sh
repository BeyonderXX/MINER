nohup python -u main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_span_test/ --epoch 30  --baseline --regular_z --regular_norm --regular_entity --regular_context --batch_size 128 --do_train --do_eval --do_predict --do_robustness_eval > bert_span_test.log 2>&1 &


nohup python main.py --gpu_id 6 --output_dir /root/RobustNER/out/bn_50_oov_1/ --gama 1 --epoch 50 --rep_mode typos --batch_size 128 --do_train --do_eval --do_predict --do_robustness_eval > bn_50_oov_1.log 2>&1 &
