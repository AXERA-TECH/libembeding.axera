#include "../src/tokenizer/tokenizer.hpp"
#include "cmdline.hpp"

int main(int argc, char *argv[])
{
    std::string tokenizer_path = "../tests/tokenizer.txt";
    cmdline::parser a;
    a.add<std::string>("tokenizer_path", 't', "tokenizer path", true);
    a.add<std::string>("text", 0, "text", true);
    a.parse_check(argc, argv);
    tokenizer_path = a.get<std::string>("tokenizer_path");

    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_path));
    auto prompt = "<s>" + a.get<std::string>("text")+"</s>";
    printf("prompt: %s\n", prompt.c_str());
    auto ids = tokenizer->encode(prompt);
    printf("ids size: %d\n", ids.size());
    for (auto id : ids)
    {
        std::cout << id << ", ";
    }
    std::cout << std::endl;

    std::string text;
    for (auto id : ids)
    {
        text += tokenizer->decode(id);
    }
    std::cout << "text: " << text << std::endl;

    return 0;
}