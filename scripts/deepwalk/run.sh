# url-sess-item
python deepwalk.py --data_file ../../dataset/prepared/i-s-u.txt --save_in_txt --output_emb_file deepwalk.txt --num_walks 10 --window_size 2 \
--walk_length 80 --lr 0.1 --negative 1 --neg_weight 1 --lap_norm 0.01 --mix --gpus 1 --num_threads 4 \
--print_interval 2000 --print_loss --batch_size 128 --use_context_weight

# item-item
python deepwalk.py --data_file ../../dataset/prepared/i-i.txt --save_in_txt --output_emb_file i-i.embed --num_walks 1000 --window_size 2 \
--walk_length 80 --lr 0.1 --negative 5 --neg_weight 1 --lap_norm 0.01 --mix --gpus 1 --num_threads 4 \
--print_interval 100 --print_loss --batch_size 128 --use_context_weight
