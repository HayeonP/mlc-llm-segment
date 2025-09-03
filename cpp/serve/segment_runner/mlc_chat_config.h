#ifndef MLC_LLM_FRONTEND_MLC_CHAT_CONFIG_H
#define MLC_LLM_FRONTEND_MLC_CHAT_CONFIG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <any>
#include <variant>

#include <picojson.h>
#include "../../json_ffi/conv_template.h"
#include "../../json_ffi/openai_api_protocol.h"
#include "../../tokenizers/tokenizers.h"

class MLCChatConfig{
public:
    MLCChatConfig(){}
    
    void FromJsonString(std::string json_string){
        picojson::value v;
        std::string err = picojson::parse(v, json_string);
        const picojson::object& obj = v.get<picojson::object>();
        
        if (obj.count("version")) {
            ICHECK(obj.at("version").is<std::string>());
            version = obj.at("version").get<std::string>();
        }
        if (obj.count("model_type")) {
            ICHECK(obj.at("model_type").is<std::string>());
            model_type = obj.at("model_type").get<std::string>();
        }
        if (obj.count("quantization")) {
            ICHECK(obj.at("quantization").is<std::string>());
            quantization = obj.at("quantization").get<std::string>();
        }
        if (obj.count("model_config")) {
            ICHECK(obj.at("model_config").is<picojson::object>());
            const picojson::object& model_config_obj = obj.at("model_config").get<picojson::object>();
            model_config = mlc::llm::json_ffi::ModelConfig::FromJSON(model_config_obj);
        }
        if (obj.count("vocab_size")) {
            ICHECK(obj.at("vocab_size").is<double>());
            vocab_size = static_cast<int>(obj.at("vocab_size").get<double>());
        }
        if (obj.count("context_window_size")) {
            ICHECK(obj.at("context_window_size").is<double>());
            context_window_size = static_cast<int>(obj.at("context_window_size").get<double>());
        }
        if (obj.count("sliding_window_size")) {
            ICHECK(obj.at("sliding_window_size").is<double>());
            sliding_window_size = static_cast<int>(obj.at("sliding_window_size").get<double>());
        }
        if (obj.count("prefill_chunk_size")) {
            ICHECK(obj.at("prefill_chunk_size").is<double>());
            prefill_chunk_size = static_cast<int>(obj.at("prefill_chunk_size").get<double>());
        }
        if (obj.count("attention_sink_size")) {
            ICHECK(obj.at("attention_sink_size").is<double>());
            attention_sink_size = static_cast<int>(obj.at("attention_sink_size").get<double>());
        }
        if (obj.count("tensor_parallel_shards")) {
            ICHECK(obj.at("tensor_parallel_shards").is<double>());
            tensor_parallel_shards = static_cast<int>(obj.at("tensor_parallel_shards").get<double>());
        }
        if (obj.count("pipeline_parallel_stages")) {
            ICHECK(obj.at("pipeline_parallel_stages").is<double>());
            pipeline_parallel_stages = static_cast<int>(obj.at("pipeline_parallel_stages").get<double>());
        }

        if (obj.count("temperature")) {
            ICHECK(obj.at("temperature").is<double>());
            temperature = obj.at("temperature").get<double>();
        }
        if (obj.count("presence_penalty")) {
            ICHECK(obj.at("presence_penalty").is<double>());
            presence_penalty = obj.at("presence_penalty").get<double>();
        }
        if (obj.count("repetition_penalty")) {
            ICHECK(obj.at("repetition_penalty").is<double>());
            repetition_penalty = obj.at("repetition_penalty").get<double>();
        }
        if (obj.count("top_p")) {
            ICHECK(obj.at("top_p").is<double>());
            top_p = obj.at("top_p").get<double>();
        }
       
        if (obj.count("tokenizer_files")) {
            ICHECK(obj.at("tokenizer_files").is<picojson::array>());
            const picojson::array& arr = obj.at("tokenizer_files").get<picojson::array>();
            for (const auto& val : arr) {
                ICHECK(val.is<std::string>());
                tokenizer_files.push_back(val.get<std::string>());
            }
        }

        if (obj.count("tokenizer_info")) {
            ICHECK(obj.at("tokenizer_info").is<picojson::object>());
            const picojson::object& tokenizer_info_obj = obj.at("tokenizer_info").get<picojson::object>();
            picojson::value tokenizer_info_val(tokenizer_info_obj);
            std::string tokenizer_info_json = tokenizer_info_val.serialize();
            tokenizer_info = mlc::llm::TokenizerInfo::FromJSONString(tokenizer_info_json);
        }
        if (obj.count("conv_template")) {
            ICHECK(obj.at("conv_template").is<picojson::object>());
            const picojson::object& conv_template_obj = obj.at("conv_template").get<picojson::object>();
            auto conv_res = mlc::llm::json_ffi::Conversation::FromJSON(conv_template_obj);
            ICHECK(conv_res.IsOk()) << "Failed to parse conv_template: " << conv_res.UnwrapErr();
            conv_template = conv_res.Unwrap();
        }

        if (obj.count("pad_token_id")) {
            ICHECK(obj.at("pad_token_id").is<double>());
            pad_token_id = static_cast<int>(obj.at("pad_token_id").get<double>());
        }
        if (obj.count("bos_token_id")) {
            ICHECK(obj.at("bos_token_id").is<double>());
            bos_token_id = static_cast<int>(obj.at("bos_token_id").get<double>());
        }

        if (obj.count("eos_token_id")) {
            ICHECK(obj.at("eos_token_id").is<picojson::array>());
            const picojson::array& arr = obj.at("eos_token_id").get<picojson::array>();
            for (const auto& val : arr) {
                ICHECK(val.is<double>());
                eos_token_id.push_back(static_cast<int>(val.get<double>()));
            }
        }
        else{
            eos_token_id.push_back(2);
        }

        return;
    }
 
    void Print() const {
        std::cout << "MLCChatConfig:" << std::endl;
        std::cout << "  version: " << version << std::endl;
        std::cout << "  model_type: " << model_type << std::endl;
        std::cout << "  quantization: " << quantization << std::endl;


        std::cout << "  model_config:" << std::endl;
        std::cout << "    vocab_size: " << model_config.vocab_size << std::endl;
        std::cout << "    context_window_size: " << model_config.context_window_size << std::endl;
        std::cout << "    sliding_window_size: " << model_config.sliding_window_size << std::endl;
        std::cout << "    prefill_chunk_size: " << model_config.prefill_chunk_size << std::endl;
        std::cout << "    tensor_parallel_shards: " << model_config.tensor_parallel_shards << std::endl;
        std::cout << "    pipeline_parallel_stages: " << model_config.pipeline_parallel_stages << std::endl;
        std::cout << "    max_batch_size: " << model_config.max_batch_size << std::endl;



        std::cout << "  vocab_size: " << vocab_size << std::endl;
        std::cout << "  context_window_size: " << context_window_size << std::endl;
        std::cout << "  sliding_window_size: " << sliding_window_size << std::endl;
        std::cout << "  prefill_chunk_size: " << prefill_chunk_size << std::endl;
        std::cout << "  attention_sink_size: " << attention_sink_size << std::endl;
        std::cout << "  tensor_parallel_shards: " << tensor_parallel_shards << std::endl;
        std::cout << "  pipeline_parallel_stages: " << pipeline_parallel_stages << std::endl;

        std::cout << "  temperature: " << temperature << std::endl;
        std::cout << "  presence_penalty: " << presence_penalty << std::endl;
        std::cout << "  frequency_penalty: " << frequency_penalty << std::endl;
        std::cout << "  repetition_penalty: " << repetition_penalty << std::endl;
        std::cout << "  top_p: " << top_p << std::endl;

        std::cout << "  tokenizer_files: [";
        for (size_t i = 0; i < tokenizer_files.size(); ++i) {
            std::cout << tokenizer_files[i];
            if (i + 1 < tokenizer_files.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  tokenizer_info:" << std::endl;
        std::cout << "     token_postproc_method: " << tokenizer_info->token_postproc_method << std::endl;
        std::cout << "     prepend_space_in_encode: " << tokenizer_info->prepend_space_in_encode << std::endl;
        std::cout << "     strip_space_in_decode: " << tokenizer_info->strip_space_in_decode << std::endl;

        std::cout << "  conv_template:" << std::endl;
        if(conv_template.name.has_value()) std::cout << "     name" << conv_template.name.value() << std::endl;
        std::cout << "     system_template" << conv_template.system_template << std::endl;
        std::cout << "     system_message" << conv_template.system_message << std::endl;
        std::cout << "     role_content_sep" << conv_template.role_content_sep << std::endl;
        std::cout << "     role_empty_sep" << conv_template.role_empty_sep << std::endl;
        for(auto v : conv_template.stop_str){
            std::cout << "     stop_str:" << std::endl;
            std::cout << "         " << v << std::endl;
        }
        for(auto v : conv_template.stop_token_ids){
            std::cout << "     stop_token_ids:" << std::endl;
            std::cout << "         " << v << std::endl;
        }

        std::cout << "  pad_token_id: " << pad_token_id << std::endl;
        std::cout << "  bos_token_id: " << bos_token_id << std::endl;

        std::cout << "  eos_token_id: [";
        for (size_t i = 0; i < eos_token_id.size(); ++i) {
            std::cout << eos_token_id[i];
            if (i + 1 < eos_token_id.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

public:
    // Version control
    std::string version = "0.1.0"; // NOTE: Hard coded
    std::string model_type = "";
    std::string quantization = "";
    mlc::llm::json_ffi::ModelConfig model_config;

    int vocab_size = -1;
    int context_window_size = -1;
    int sliding_window_size = -1;
    int prefill_chunk_size = -1;
    int attention_sink_size = -1;
    int tensor_parallel_shards = -1;
    int pipeline_parallel_stages = 1;
    // Configuration of text generation
    double temperature = 1.0;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    double top_p = 1.0;
    // Tokenizer configuration
    std::vector<std::string> tokenizer_files;
    // The content of tokenizer.TokenizerInfo
    mlc::llm::TokenizerInfo tokenizer_info;
    // conversation template
    mlc::llm::json_ffi::Conversation conv_template;
    // extra fields from generation_config.json
    // NOTE: they are not being used for now in MLCEngine
    // but we keep them for book-keep purposes
    int pad_token_id = 0;
    int bos_token_id = 1;
    std::vector<int> eos_token_id;
};

#endif