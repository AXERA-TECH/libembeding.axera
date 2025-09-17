#ifndef __EMBEDING_H__
#define __EMBEDING_H__

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        char template_filename_axmodel[1024];
        int axmodel_num;

        char url_tokenizer_model[1024];

        char filename_tokens_embed[1024];
        int tokens_embed_num;
        int tokens_embed_size;

    } embeding_attr_t;

    typedef struct
    {
        float embeding[1024];
    } embeding_t;

    typedef void *embeding_handle_t;

    int ax_embeding_init(embeding_attr_t *attr, embeding_handle_t *handle);
    int ax_embeding_deinit(embeding_handle_t handle);

    int ax_embeding(embeding_handle_t handle, char *text, embeding_t *embeding);

#ifdef __cplusplus
}
#endif

#endif
