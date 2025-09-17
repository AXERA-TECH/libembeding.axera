#include "../include/embeding.h"

#include "cmdline.hpp"
#include "timer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>
#include <cmath>

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
    embeding_attr_t attr;

    cmdline::parser cmd;
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("url_tokenizer_model", 0, "tokenizer model path", false, attr.url_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    cmd.parse_check(argc, argv);

    strcpy(attr.url_tokenizer_model, cmd.get<std::string>("url_tokenizer_model").c_str());

    attr.axmodel_num = cmd.get<int>("axmodel_num");
    strcpy(attr.template_filename_axmodel, cmd.get<std::string>("template_filename_axmodel").c_str());

    attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");
    strcpy(attr.filename_tokens_embed, cmd.get<std::string>("filename_tokens_embed").c_str());

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

    embeding_handle_t handle;
    ret = ax_embeding_init(&attr, &handle);
    if (0 != ret)
    {
        printf("ax_embeding_init failed\n");
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }

    timer t;

    std::string task = "Given a web search query, retrieve relevant passages that answer the query";
    std::vector<std::string> queries = {
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity")};

    std::vector<embeding_t> queries_embeddings(queries.size());
    for (int i = 0; i < queries.size(); i++)
    {
        t.start();
        ax_embeding(handle, (char *)queries[i].c_str(), &queries_embeddings[i]);
        t.stop();
        printf("time: %5.2f\n", t.cost());
        printf("%5.2f %5.2f %5.2f ... %5.2f %5.2f %5.2f\n",
               queries_embeddings[i].embeding[0], queries_embeddings[i].embeding[1], queries_embeddings[i].embeding[2],
               queries_embeddings[i].embeding[1021], queries_embeddings[i].embeding[1022], queries_embeddings[i].embeding[1023]);
    }

    std::vector<std::string> documents = {
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."};
    std::vector<embeding_t> documents_embeddings(documents.size());
    for (int i = 0; i < documents.size(); i++)
    {
        t.start();
        ax_embeding(handle, (char *)documents[i].c_str(), &documents_embeddings[i]);
        t.stop();
        printf("time: %5.2f\n", t.cost());
        printf("%5.2f %5.2f %5.2f ... %5.2f %5.2f %5.2f\n",
               documents_embeddings[i].embeding[0], documents_embeddings[i].embeding[1], documents_embeddings[i].embeding[2],
               documents_embeddings[i].embeding[1021], documents_embeddings[i].embeding[1022], documents_embeddings[i].embeding[1023]);
    }

    // 0.75 ,0.15, 0.11, 0.61
    for (int i = 0; i < queries_embeddings.size(); i++)
    {
        for (int j = 0; j < documents_embeddings.size(); j++)
        {
            std::vector<float> q(queries_embeddings[i].embeding, queries_embeddings[i].embeding + 1024);
            std::vector<float> d(documents_embeddings[j].embeding, documents_embeddings[j].embeding + 1024);
            q = l2norm(q);
            d = l2norm(d);
            float similarity = compare(q.data(), d.data(), q.size());
            printf("similarity: %5.2f \n", similarity);
        }
    }

    ax_embeding_deinit(handle);

    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}