#include "embeding.h"

#include "LLM.hpp"

struct embeding_handle_internal_t
{
    LLMAttrType attr;
    LLM lLaMa;
};

int ax_embeding_init(embeding_attr_t *attr, embeding_handle_t *handle)
{
    embeding_handle_internal_t *internal = new embeding_handle_internal_t();

    internal->attr.url_tokenizer_model = attr->url_tokenizer_model;

    internal->attr.axmodel_num = attr->axmodel_num;
    internal->attr.template_filename_axmodel = attr->template_filename_axmodel;

    internal->attr.filename_tokens_embed = attr->filename_tokens_embed;
    internal->attr.tokens_embed_num = attr->tokens_embed_num;
    internal->attr.tokens_embed_size = attr->tokens_embed_size;

    internal->attr.b_use_mmap_load_embed = 1;

    if (!internal->lLaMa.Init(internal->attr))
    {
        return -1;
    }
    *handle = internal;

    return 0;
}

int ax_embeding_deinit(embeding_handle_t handle)
{
    if (handle == nullptr)
    {
        return 0;
    }
    embeding_handle_internal_t *internal = (embeding_handle_internal_t *)handle;
    internal->lLaMa.Deinit();
    delete internal;
    return 0;
}

int ax_embeding(embeding_handle_t handle, char *text, embeding_t *embeding)
{
    if (embeding == nullptr)
    {
        return -1;
    }
    embeding_handle_internal_t *internal = (embeding_handle_internal_t *)handle;

    std::vector<int> _token_ids;
    internal->lLaMa.Encode(_token_ids, text);
    std::vector<float> embed;
    internal->lLaMa.GenerateEmbedingPrefill(_token_ids, embed);

    if (embed.size() != 1024)
    {
        ALOGE("embed.size() != 1024");
        return -1;
    }

    memcpy(embeding->embeding, embed.data(), embed.size() * sizeof(float));
    return 0;
}