# libembeding.axera
text embeding sdk base Qwen/Qwen3-Embedding-0.6B run in AXERA CHIP

## get models
[AXERA-TECH/Qwen3-Embedding-0.6B](https://huggingface.co/AXERA-TECH/Qwen3-Embedding-0.6B)

```shell
export HF_ENDPOINT="https://hf-mirror.com" # optional for China mainland
hf download AXERA-TECH/Qwen3-Embedding-0.6B --local-dir Qwen3-Embedding-0.6B
```

## Build
```shell
sh ./build.sh
```

### Run
```shell
./run_qwen3_0.6B_embeding.sh 
[I][                            Init][  70]: LLM init start
tokenizer_type = 3
100% | ███████████████████████████████ |  30 /  30 [4.60s<4.60s, 6.52 count/s] init 27 axmodel ok,remain_cmm(1851 MB)
[I][     Init][ 141]: max_token_len : 2559
[I][     Init][ 146]: kv_cache_size : 1024, kv_cache_num: 2559
[I][     Init][ 154]: prefill_token_num : 128
[I][     Init][ 158]: grp: 1, prefill_max_token_num : 1
[I][     Init][ 158]: grp: 2, prefill_max_token_num : 128
[I][     Init][ 158]: grp: 3, prefill_max_token_num : 256
[I][     Init][ 158]: grp: 4, prefill_max_token_num : 384
[I][     Init][ 158]: grp: 5, prefill_max_token_num : 512
[I][     Init][ 158]: grp: 6, prefill_max_token_num : 640
[I][     Init][ 158]: grp: 7, prefill_max_token_num : 768
[I][     Init][ 158]: grp: 8, prefill_max_token_num : 896
[I][     Init][ 158]: grp: 9, prefill_max_token_num : 1024
[I][     Init][ 162]: prefill_max_token_num : 1024
[I][     Init][ 166]: LLM init ok
time:
-10.19 -0.89 -6.00 ... 17.25  8.00 -2.64
time: 318.40
-2.47 -1.08  0.00 ... -6.41 -0.03  1.31
time: 288.64
-11.81 -0.57 -36.00 ... 16.38 20.38 -4.41
time: 340.39
-13.81 -0.61 12.00 ...  1.77 -6.06  5.22
similarity:  0.75 
similarity:  0.15 
similarity:  0.11 
similarity:  0.61 
```

## Reference
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

[MNN-LLM](https://github.com/wangzhaode/mnn-llm)



## 技术讨论

- Github issues
- QQ 群: 139953715
