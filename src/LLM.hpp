#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "utils/bfloat16.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "utils/ax_cmm_utils.hpp"
#include "utils/cqdm.h"
#include "tokenizer/tokenizer.hpp"
#
#include <ax_sys_api.h>

#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 28;

    int prefill_token_num = 128; // auto calc
    int prefill_max_token_num = 1024;
    std::vector<int> prefill_max_kv_cache_num_grp;

    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 151669;
    int tokens_embed_size = 1024;

    std::string url_tokenizer_model;

    int max_token_len = 2559; // auto calc

    int kv_cache_num = 2559;  // auto calc
    int kv_cache_size = 1024; // auto calc

    // int precompute_len = 1202;

    // int prefill_grpid = -1;

    bool b_use_mmap_load_embed = false;
};

class LLM
{
private:
    LLaMaEmbedSelector embed_selector;
    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer;

    LLMAttrType _attr;

    struct LLMLayer
    {
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    // ax_runner_ax650 llama_post;

    int decode_grpid = 0;

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 2, 31);
        this->_attr = attr;
        tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(attr.url_tokenizer_model));
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");
        // test code
        // {
        //     std::vector<int> output;
        //     tokenizer.Encode("Today is National", output);
        //     // print output
        //     for (size_t i = 0; i < output.size(); i++)
        //     {
        //         printf("%d ", output[i]);
        //     }
        //     printf("\n");
        // }

        if (!embed_selector.Init(attr.filename_tokens_embed, attr.tokens_embed_num, attr.tokens_embed_size, attr.b_use_mmap_load_embed))
        {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.tokens_embed_num, attr.tokens_embed_size);
            return false;
        }
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");
        // test code
        // {
        //     std::vector<unsigned short> embed = embed_selector.getByIndex(123);
        //     printf("embed size: %d\n", embed.size());
        //     for (int i = 0; i < embed.size(); i++)
        //     {
        //         bfloat16 bf16 = bfloat16(embed[i]);
        //         float val = bf16;
        //         printf("%d %0.22f\n", embed[i], val);
        //     }
        // }

        llama_layers.resize(attr.axmodel_num);
        // prefill_layers.resize(attr.prefill_axmodel_num);

        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), false);
            if (ret != 0)
            {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, i + 2, "count", axmodel_path);
        }
        printf("\n");
        {   
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            ALOGI("max_token_len : %d", _attr.max_token_len);
            // auto &input_k_cache = llama_layers[0].layer.get_input("K_cache");
            // auto &output_k_cache_out = llama_layers[0].layer.get_output("K_cache_out");
            _attr.kv_cache_size = llama_layers[0].layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num = llama_layers[0].layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num)
            {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }

            _attr.prefill_token_num = llama_layers[0].layer.get_input(1, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
            for (size_t i = 0; i < llama_layers[0].layer.get_num_input_groups() - 1; i++)
            {
                int prefill_max_kv_cache_num = llama_layers[0].layer.get_input(i + 1, "K_cache").vShape[1];
                ALOGI("grp: %d, prefill_max_token_num : %d", i + 1, prefill_max_kv_cache_num);
                _attr.prefill_max_kv_cache_num_grp.push_back(prefill_max_kv_cache_num);
            }
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        }

        // Reset();
        ALOGI("LLM init ok");
        return true;
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            llama_layers[i].layer.release();
        }
        // llama_post.release();
        embed_selector.Deinit();
    }

    int Encode(std::vector<int> &_token_ids, std::string prompt)
    {
        _token_ids = tokenizer->encode(prompt);
        _token_ids.push_back(151643);

        return 0;
    }

    int GenerateEmbedingPrefill(std::vector<int> &_token_ids, std::vector<float> &out_embed)
    {
        bfloat16 bf16 = -65536.f;
        int input_embed_num = _token_ids.size();

        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);

        int prefill_grpid = _attr.prefill_max_kv_cache_num_grp.size();

        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (input_embed_num <= _attr.prefill_max_kv_cache_num_grp[i])
            {
                prefill_grpid = i + 1;
                break;
            }
        }
        // ALOGI("input token num : %d, prefill_split_num : %d prefill_grpid : %d", input_embed_num, prefill_split_num, prefill_grpid);

        // clear kv cache
        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            memset((void *)llama_layers[i].layer.get_input(prefill_grpid, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(prefill_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(prefill_grpid, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(prefill_grpid, "V_cache").nSize);
        }

        if (input_embed_num == 0)
        {
            ALOGI("input token num is 0, skip");
            return 0;
        }

        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[prefill_grpid - 1];

        std::vector<unsigned short> test_embed, test_embed_out;
        test_embed.resize(_token_ids.size() * _attr.tokens_embed_size);
        test_embed_out.resize(_attr.tokens_embed_size);

        for (size_t i = 0; i < _token_ids.size(); i++)
        {
            embed_selector.getByIndex(_token_ids[i], test_embed.data() + i * _attr.tokens_embed_size);
        }
        out_embed.resize(_attr.tokens_embed_size);

        for (size_t p = 0; p < prefill_split_num; p++)
        {
            std::vector<unsigned short> mask_tmp;
            mask_tmp.resize(1 * _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            int input_num_token = _attr.prefill_token_num;
            if (p == prefill_split_num - 1)
            {
                input_num_token = input_embed_num - p * _attr.prefill_token_num;
            }

            // ALOGI("input_num_token: %d", input_num_token);
            for (size_t i = 0; i < _attr.prefill_token_num; i++)
            {
                if (i < input_num_token)
                {
                    int mask_current_start = kv_cache_num;
                    auto mask_ptr = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);

                    for (int j = 0; j < p * _attr.prefill_token_num; j++)
                    {
                        mask_ptr[j] = 0;
                    }

                    for (int j = mask_current_start; j < mask_current_start + i + 1; j++)
                    {
                        mask_ptr[j] = 0;
                    }
                }
            }

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short));
            }
            else
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++)
            {
                auto &layer = llama_layers[m];
                // set indices
                auto &input_indices = layer.layer.get_input(prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                int idx = 0;
                for (unsigned int i = p * _attr.prefill_token_num; i < (p + 1) * _attr.prefill_token_num; i++)
                {
                    input_indices_ptr[idx] = i;
                    idx++;
                }
                // memcpy((void *)input_indices.phyAddr, input_indices_ptr, input_indices.nSize);

                // set mask
                auto &input_mask = layer.layer.get_input(prefill_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input(prefill_grpid, "input");
                memcpy((void *)input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer.inference(prefill_grpid);

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_prefill_k_cache = layer.layer.get_input(prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer.get_input(prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(prefill_grpid, "V_cache_out");

                int kv_offset = (p * _attr.prefill_token_num) * _attr.kv_cache_size;

                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(prefill_grpid, "output");
                memcpy(embed_tmp.data(), (void *)output.pVirAddr, embed_tmp.size() * sizeof(unsigned short));

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            if (p == (prefill_split_num - 1))
            {
                memcpy(test_embed_out.data(),
                       embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size,
                       _attr.tokens_embed_size * sizeof(unsigned short));
            }
        }
        out_embed.resize(_attr.tokens_embed_size);
        for (int i = 0; i < _attr.tokens_embed_size; i++)
        {
            out_embed[i] = bfloat16(test_embed_out[i]).fp32();
        }

        return 0;
    }
};
