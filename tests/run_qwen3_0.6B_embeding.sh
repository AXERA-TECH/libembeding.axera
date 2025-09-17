./test_api \
--template_filename_axmodel "Qwen3-Embedding-0.6B/qwen3_embedding_0.6b_axmodel/qwen3_p128_l%d_together.axmodel" \
--axmodel_num 28 \
--url_tokenizer_model "../tests/tokenizer.txt" \
--filename_tokens_embed "Qwen3-Embedding-0.6B/qwen3_embedding_0.6b_axmodel/model.embed_tokens.weight.bfloat16.bin" \
--tokens_embed_num 151669 \
--tokens_embed_size 1024
