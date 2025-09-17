#include "../src/LLM.hpp"

#include "cmdline.hpp"
#include <signal.h>

#include <ax_sys_api.h>
#include <ax_engine_api.h>

static LLM lLaMa;

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    fprintf(stdout, "%s", p_str);
    fflush(stdout);
}

std::vector<float> l2norm(std::vector<float> embedding)
{
    float norm = 0.0f;
    for (int j = 0; j < embedding.size(); j++)
    {
        norm += embedding[j] * embedding[j];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-12f)
    {
        for (int j = 0; j < embedding.size(); j++)
        {
            embedding[j] /= norm;
        }
    }
    return embedding;
}

float compare(float *a, float *b, int len)
{

    float similarity = 0.0;
    for (int i = 0; i < len; i++)
        similarity += a[i] * b[i];
    similarity = similarity < 0 ? 0 : similarity > 1 ? 1
                                                     : similarity;
    return similarity;
}

std::string get_detailed_instruct(const std::string &task_description, const std::string &query)
{
    return "Instruct: " + task_description + "\nQuery: " + query;
}

int main(int argc, char *argv[])
{
    LLMAttrType attr;

    cmdline::parser cmd;
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("url_tokenizer_model", 0, "tokenizer model path", false, attr.url_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);

    cmd.parse_check(argc, argv);

    attr.url_tokenizer_model = cmd.get<std::string>("url_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");

    attr.axmodel_num = cmd.get<int>("axmodel_num");
    attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");

    attr.b_use_mmap_load_embed = cmd.get<bool>("use_mmap_load_embed");

    // 1. init engine
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_SYS_Init();
    auto ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret)
    {
        return ret;
    }

    if (!lLaMa.Init(attr))
    {
        ALOGE("lLaMa.Init failed");
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }

    std::string task = "Given a web search query, retrieve relevant passages that answer the query";
    std::vector<std::string> queries = {
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity")};

    std::vector<std::vector<float>> queries_embeddings;
    for (int i = 0; i < queries.size(); i++)
    {
        std::vector<int> _token_ids;
        lLaMa.Encode(_token_ids, queries[i]);
        std::cout << "query: " << queries[i] << std::endl;
        for (auto id : _token_ids)
        {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        std::vector<float> embed;
        lLaMa.GenerateEmbedingPrefill(_token_ids, embed);
        //
        queries_embeddings.push_back(embed);
        printf("%5.2f %5.2f %5.2f ... %5.2f %5.2f %5.2f\n", embed[0], embed[1], embed[2], embed[1021], embed[1022], embed[1023]);
    }

    std::vector<std::string> documents = {
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."};
    std::vector<std::vector<float>> documents_embeddings;
    for (int i = 0; i < documents.size(); i++)
    {
        std::vector<int> _token_ids;
        lLaMa.Encode(_token_ids, documents[i]);
        std::cout << "document: " << documents[i] << std::endl;
        for (auto id : _token_ids)
        {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        std::vector<float> embed;
        lLaMa.GenerateEmbedingPrefill(_token_ids, embed);
        documents_embeddings.push_back(embed);
        printf("%5.2f %5.2f %5.2f ... %5.2f %5.2f %5.2f\n", embed[0], embed[1], embed[2], embed[1021], embed[1022], embed[1023]);
    }

    // 0.75 ,0.15, 0.11, 0.61
    for (int i = 0; i < queries_embeddings.size(); i++)
    {
        for (int j = 0; j < documents_embeddings.size(); j++)
        {
            auto q = l2norm(queries_embeddings[i]);
            auto d = l2norm(documents_embeddings[j]);
            float similarity = compare(q.data(), d.data(), q.size());
            printf("similarity: %5.2f \n", similarity);
        }
    }

    lLaMa.Deinit();

    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}