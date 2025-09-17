#include "../src/tokenizer/tokenizer.hpp"
#include "cmdline.hpp"

std::string get_detailed_instruct(const std::string &task_description, const std::string &query)
{
    return "Instruct: " + task_description + "\nQuery: " + query;
}

enum PaddingSide
{
    LEFT,
    RIGHT
};

std::vector<int> pad_ids(std::vector<int> ids, int len, int pad_token_id, PaddingSide padding_side = PaddingSide::LEFT)
{
    ids.push_back(pad_token_id);
    int len_ids = ids.size();
    if (len - len_ids > 0)
    {
        switch (padding_side)
        {
        case PaddingSide::LEFT:
            for (int i = 0; i < len - len_ids; ++i)
            {
                ids.insert(ids.begin(), pad_token_id);
            }
            break;
        case PaddingSide::RIGHT:
            for (int i = 0; i < len - len_ids; ++i)
            {
                ids.push_back(pad_token_id);
            }
            break;
        default:
            break;
        }
    }

    return ids;
}

int main(int argc, char *argv[])
{
    std::string tokenizer_path = "../tests/tokenizer.txt";
    cmdline::parser a;
    a.add<std::string>("tokenizer_path", 't', "tokenizer path", true);
    a.parse_check(argc, argv);
    tokenizer_path = a.get<std::string>("tokenizer_path");

    /*
        // Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?
        [[151643, 151643, 151643, 151643,    641,   1235,     25,  16246,    264,
           3482,   2711,   3239,     11,  17179,   9760,  46769,    429,   4226,
            279,   3239,    198,   2859,     25,   3838,    374,    279,   6722,
            315,   5616,     30, 151643],

        // Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:Explain gravity
        [151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643,    641,
           1235,     25,  16246,    264,   3482,   2711,   3239,     11,  17179,
           9760,  46769,    429,   4226,    279,   3239,    198,   2859,     25,
            840,  20772,  23249, 151643],

        // The capital of China is Beijing.
        [151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643,
         151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643,
         151643, 151643, 151643, 151643, 151643,    785,   6722,    315,   5616,
            374,  26549,     13, 151643],

        // Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.
        [ 38409,    374,    264,   5344,    429,  60091,   1378,  12866,   6974,
           1817,   1008,     13,   1084,   6696,   4680,    311,   6961,   6171,
            323,    374,   8480,    369,    279,   7203,    315,  32875,   2163,
            279,   7015,     13, 151643]]

    */

    std::string task = "Given a web search query, retrieve relevant passages that answer the query";

    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_path));

    std::vector<std::string> queries = {
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity")};
    for (auto &query : queries)
    {
        std::cout << query << std::endl;
        auto ids = tokenizer->encode(query);
        ids = pad_ids(ids, 31, 151643, PaddingSide::LEFT);
        printf("ids size: %d\n", ids.size());
        for (auto id : ids)
        {
            std::cout << id << " ";
        }
        std::cout << std::endl;
    }

    std::vector<std::string> documents = {
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."};
    for (auto &document : documents)
    {
        std::cout << document << std::endl;
        auto ids = tokenizer->encode(document);
        ids = pad_ids(ids, 31, 151643, PaddingSide::LEFT);
        printf("ids size: %d\n", ids.size());
        for (auto id : ids)
        {
            std::cout << id << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}