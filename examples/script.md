# baseline

nohup python run_crf_ner_baseline.py > bert_bsl_origin.log 2>&1 &


nohup python run_crf_ner.py > bert_bsl_test.log 2>&1 &
17046

nohup python run_crf_ner_bn.py > bert_bn.log 2>&1 &
26197

nohup python run_crf_ner_bn_2.py > bert_bn_2.log 2>&1 &
26291


nohup python run_crf_ner_bn_wo_reg.py > bert_bn_wo_reg.log 2>&1 &


nohup python run_crf_ner_bn_2.py > bert_bn_5e_7.log 2>&1 &

# 测试 baseline
--gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/baseline --do_train --do_eval --do_predict --baseline --do_robustness_eval

# 测试无 regular z
--gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/wo_reg_z --do_train --do_eval --do_predict --regular_z --do_robustness_eval

# 测试 beta 参数设置
--gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_7 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-7
