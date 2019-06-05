set DATA_DIR=D:\project\bert\data5
set BERT_BASE_DIR=D:\Downloads\chinese_L-12_H-768_A-12
set OUT_DIR=d:\temp\out5
set EXPORT_DIR=D:\project\bert\export
python run_classifier.py  --task_name=dg  --do_train=true  --do_eval=false --do_predict=false --do_export=false --data_dir=%DATA_DIR%\  --vocab_file=%BERT_BASE_DIR%\vocab.txt  --bert_config_file=%BERT_BASE_DIR%\bert_config.json  --init_checkpoint=%BERT_BASE_DIR%\bert_model.ckpt  --output_dir=%OUT_DIR% --export_dir=%EXPORT_DIR% --train_batch_size=32