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



# hidden dim 实验
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h200 --hidden_dim 200 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h200.log 2>&1 &
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h100 --hidden_dim 100 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h100.log 2>&1 &
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h50 --hidden_dim 50 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h50.log 2>&1 &

nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h500 --hidden_dim 500 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h500.log 2>&1 &
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bert_uncase/beta_5e_6_h756 --hidden_dim 756 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-6 > bert_beta_5e_6_h756.log 2>&1 &


# 第一版entity regularizer实验（gamma 1e-03  total_typos）
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3/ --batch_size 32 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bert_beta_5e_5_h300_ent_typos.log 2>&1 &
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3_h_30/ --batch_size 32 --hidden_dim 30 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bert_beta_5e_5_h30_ent_typos.log 2>&1 &

# gama 实验
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_gama_1e_4_h_300/ --batch_size 32 --hidden_dim 300 --gama 1e-4 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bert_beta_5e_5_gama_1e_4_h300_ent_typos.log 2>&1 &

# 变形实验
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3_ngram/ --rep_mode ngram --batch_size 32 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bert_beta_5e_5_ngram_ent_typos.log 2>&1 &


# 变分 KL散度项 移除实验
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3_wo_reg_z_typos/ --regular_z --batch_size 40 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bn_ent_typos_1e_3_wo_reg_z_typos.log 2>&1 &
48897
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3_wo_reg_z_ngram/ --regular_z --rep_mode ngram --batch_size 40 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bn_ent_typos_1e_3_wo_reg_z_ngram.log 2>&1 &
48977
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3_wo_reg_z_typos_sample_1/ --sample_size 1 --regular_z --batch_size 40 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bn_ent_typos_1e_3_wo_reg_z_typos_sample_1.log 2>&1 &
49057

# 移除 entity MI 实验
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3_wo_reg_ent/ --regular_entity --batch_size 40 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bn_ent_typos_1e_3_wo_reg_ent.log 2>&1 &

nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3_wo_reg_z_reg_ent/ --regular_entity --regular_z --batch_size 40 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bn_ent_typos_1e_3_wo_reg_z_typos.log 2>&1 &

nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3_wo_reg_z_reg_ent_sample_1/ --sample_size 1 --regular_entity --regular_z --batch_size 40 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5 > bn_ent_typos_1e_3_wo_reg_z_reg_ent_sample_1.log 2>&1 &

# 大参数 debug 设置
--gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_typos_1e_3_wo_reg_z_reg_ent_sample_1/ --epoch 10 --sample_size 1 --regular_entity --regular_z --batch_size 40 --hidden_dim 300 --do_train --do_eval --do_predict --do_robustness_eval --beta 5e-5



# 不加 norm 约束， 不加 I(x;z)约束，只有 loglikely + entity reg
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/bn_ent_reg_sample1/ --sample_size 1 --epoch 50 --regular_z --regular_norm --batch_size 36 --do_train --do_eval --do_predict --do_robustness_eval > bn_ent_reg_sample1.log 2>&1 &

# 加 norm 约束， 不加 I(x;z)约束，有 loglikely + norm + entity reg , sample size 1
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/bn_norm_ent_reg_sample1/ --sample_size 1 --epoch 50 --regular_z --batch_size 36 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample1.log 2>&1 &

# 加 norm 约束， 不加 I(x;z)约束，有 loglikely + norm + entity reg , sample size 5
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/bn_norm_ent_reg_sample5/ --sample_size 5 --epoch 50 --regular_z --batch_size 36 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5.log 2>&1 &


# relu 版激活函数(loss跑飞)
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bert_uncase/bn_relu_norm_ent_reg_sample5/ --sample_size 5 --epoch 40 --regular_z --batch_size 36 --do_train --do_eval --do_predict --do_robustness_eval > bn_relu_norm_ent_reg_sample5.log 2>&1 &

# sample size 实验(tanh) （鲁棒性下降）
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bert_uncase/bn_norm_ent_reg_sample10/ --sample_size 10 --epoch 50 --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample10.log 2>&1 &
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bert_uncase/bn_norm_ent_reg_sample15/ --sample_size 15 --epoch 50 --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample15.log 2>&1 &

# 22服务器
# V0 evaluate
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bn_baseline/ --baseline --regular_z --batch_size 32 --do_predict --do_robustness_eval > bn_baseline_v0.log 2>&1 &
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample1/ --regular_z --batch_size 32 --do_predict --do_robustness_eval > bn_norm_ent_reg_sample1_v0.log 2>&1 &
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5/ --regular_z --batch_size 32 --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_v0.log 2>&1 &

# 增大entity惩罚力度
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_gama_5e_3/ --epoch 40 --gama 5e-3 --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_gama_5e_3.log 2>&1 &
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_gama_1e_2/ --epoch 40 --gama 1e-2 --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_gama_1e_2.log 2>&1 &
nohup python main.py --gpu_id 5 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_gama_1e_1/ --epoch 40 --gama 1e-1 --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_gama_1e_1.log 2>&1 &
nohup python main.py --gpu_id 6 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_gama_1/ --epoch 40 --gama 1 --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_gama_1.log 2>&1 &
nohup python main.py --gpu_id 7 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_gama_5/ --epoch 40 --gama 5 --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_gama_5.log 2>&1 &

# 加入 pmi 过滤，epoch negative sample构建策略
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi_ngram/ --epoch 40 --rep_total --pmi_preserve 0.2 --rep_mode ngram --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi_ngram.log 2>&1 &

nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi_typos/ --epoch 40 --rep_total --pmi_preserve 0.2 --rep_mode typos --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi_typos.log 2>&1 &

# pmi ratio 实验
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi1_typos/ --epoch 50 --rep_total --pmi_preserve 0.1 --rep_mode typos --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi1_typos.log 2>&1 &
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi3_typos/ --epoch 50 --rep_total --pmi_preserve 0.3 --rep_mode typos --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi3_typos.log 2>&1 &
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi4_typos/ --epoch 50 --rep_total --pmi_preserve 0.4 --rep_mode typos --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi4_typos.log 2>&1 &

nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi1_ngram/ --epoch 50 --rep_total --pmi_preserve 0.1 --rep_mode ngram --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi1_ngram.log 2>&1 &
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi3_ngram/ --epoch 50 --rep_total --pmi_preserve 0.3 --rep_mode ngram --regular_z --batch_size 20 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi3_typos.log 2>&1 &
# nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi4_ngram/ --epoch 50 --rep_total --pmi_preserve 0.4 --rep_mode ngram --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi4_typos.log 2>&1 &


# pmi 策略改进实验
# 如果将替换词 tokenize ，那其数据处理方式和测试时候不同，导致对比无效
--gpu_id 1  --epoch 5 --rep_total --pmi_preserve 0.3 --rep_mode typos --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval
--gpu_id 1  --epoch 5 --rep_total --pmi_preserve 0.3 --rep_mode ngram --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval
--gpu_id 1  --epoch 5 --rep_total --pmi_preserve 0.3 --rep_mode ngram --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval

nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi2_typos_new/ --epoch 50 --rep_total --pmi_preserve 0.2 --rep_mode typos --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi2_typos_new.log 2>&1 &
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi3_typos_new/ --epoch 50 --rep_total --pmi_preserve 0.3 --rep_mode typos --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi3_typos_new.log 2>&1 &
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi35_typos_new/ --epoch 50 --rep_total --pmi_preserve 0.35 --rep_mode typos --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi35_typos_new.log 2>&1 &

nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi2_ngram_new/ --epoch 50 --rep_total --pmi_preserve 0.2 --rep_mode ngram --regular_z --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi2_ngram_new.log 2>&1 &
nohup python main.py --gpu_id 0 --output_dir /root/RobustNER/out/bn_norm_ent_reg_sample5_pmi3_ngram_new/ --epoch 50 --rep_total --pmi_preserve 0.3 --rep_mode ngram --regular_z --batch_size 20 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_ent_reg_sample5_pmi3_ngram_new.log 2>&1 &



# fix 模型参数实验
# 只 压缩最后一层，可以finetuning的参数太少，最终欠拟合
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_norm_fix/ --epoch 50 --rep_total --pmi_preserve 0.3 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_fix.log 2>&1 &

# 多层 embedding 融合实验
# fix 模型
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bn_norm_fix_multi/ --epoch 50 --rep_total --pmi_preserve 0.3 --rep_mode typos --batch_size 128 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_fix_multi.log 2>&1 &
# 不fix模型
nohup python main.py --gpu_id 3 --output_dir /root/RobustNER/out/bn_norm_multi/ --epoch 50  --fix_bert --rep_total --pmi_preserve 0.3 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_multi.log 2>&1 &
nohup python main.py --gpu_id 4 --output_dir /root/RobustNER/out/bn_norm_1/ --epoch 50  --fix_bert --multi_layers --rep_total --pmi_preserve 0.3 --rep_mode typos --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bn_norm_1.log 2>&1 &

# 简单加入context, OOV 提升5个点
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bsl_context/ --epoch 30  --baseline --multi_layers --fix_bert --regular_entity --regular_z --regular_norm --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bsl_context.log 2>&1 &


# InfoNCE loss 实验
nohup python main.py --gpu_id 1 --output_dir /root/RobustNER/out/bsl_reg_con/ --epoch 20  --multi_layers --fix_bert --regular_entity --regular_z --regular_norm --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bsl_reg_con.log 2>&1 &
nohup python main.py --gpu_id 2 --output_dir /root/RobustNER/out/bsl_reg_con_entity/ --epoch 20  --multi_layers --fix_bert --regular_z --regular_norm --batch_size 32 --do_train --do_eval --do_predict --do_robustness_eval > bsl_reg_con_entity.log 2>&1 &

