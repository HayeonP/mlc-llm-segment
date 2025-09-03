#ifndef MLC_LLM_SERVE_SEGMENT_RUNNER_SEGMENT_RUNNER_H_
#define MLC_LLM_SERVE_SEGMENT_RUNNER_SEGMENT_RUNNER_H_

#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>
#include <variant>
#include <thread>
#include <any>
#include <algorithm>
#include <stop_token>

#include <picojson.h>

#include "../../serve/config.h"
#include "../../serve/threaded_engine.h"
#include "../../serve/data.h"
#include "../../serve/config.h"
#include "../../serve/request.h"
#include "../../tokenizers/tokenizers.h"
#include "../../tokenizers/streamer.h"
#include "../../json_ffi/conv_template.h"
#include "../../json_ffi/openai_api_protocol.h"

#include <tvm/runtime/device_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/any.h>
#include <tvm/runtime/int_tuple.h>

#include <stdexcept>
#include <csignal>
#include <atomic>

#include "./utils.h"
#include "./blocking_queue.h"
#include "./scope_fail.h"
#include "./generator.h"
#include "./data_types.h"
#include "./engine_base.h"
#include "./mlc_chat_config.h"

using namespace tvm;
using namespace ffi;

using TokenIds = IntTuple; // tvm::ffi::Shape
using String = tvm::ffi::String;

using ModelArg = std::unordered_map<std::string, std::string>;
using Conversation = mlc::llm::json_ffi::Conversation;
using ChatCompletionRequest = mlc::llm::json_ffi::ChatCompletionRequest;
using ChatCompletionMessage = mlc::llm::json_ffi::ChatCompletionMessage;
using ChatCompletionMessageContent = mlc::llm::json_ffi::ChatCompletionMessageContent;
using ChatCompletionStreamResponse = mlc::llm::json_ffi::ChatCompletionStreamResponse;
using ChatCompletionStreamResponseChoice = mlc::llm::json_ffi::ChatCompletionStreamResponseChoice;
using ChatCompletionResponse = mlc::llm::json_ffi::ChatCompletionResponse;
using ChatCompletionResponseChoice = mlc::llm::json_ffi::ChatCompletionResponseChoice;

class SegmentRunner {
public:
  SegmentRunner() {}
  ~SegmentRunner(){}
  void Init(std::string model, tvm::Device& device, std::string model_lib, std::string mode);
  void Request(std::string& prompt, int max_tokens);
  std::string Execute();
  bool IsEnd();
  
private:
  void _check_engine_config(std::string model, std::string model_lib, EngineMode mode, mlc::llm::serve::EngineConfig engine_config);
  std::vector<ModelInfo> _parse_members(std::string model, std::string model_lib);
  void _convert_model_info(ModelInfo model, Conversation& conversation, std::vector<std::string>& config_file_paths, std::string& output_model_path, std::string& output_model_lib);
  std::vector<ModelInfo> _parse_models(std::string model, std::string model_lib);
  void _process_model_args(std::vector<ModelInfo>& models, tvm::Device& device, mlc::llm::serve::EngineConfig& engine_config, std::vector<ModelArg>& output_model_args, std::vector<std::string>& output_config_file_paths, Conversation& output_conv_template);
  void _sync_request_stream_callback(tvm::ffi::Array<mlc::llm::serve::RequestStreamOutput> delta_outputs);

  Generator<std::vector<CallbackStreamOutput>> _generate_segment_output();
  void _request_stream_callback_impl(std::vector<mlc::llm::serve::RequestStreamOutput> delta_outputs, std::vector<std::vector<CallbackStreamOutput>>& output_request_outputs, Optional<String>& output_request_final_usage_json_str);
  ChatCompletionRequest _create_chat_completion_request(std::string& model, std::string prompt, int max_tokens, bool stream);
  ChatCompletionResponse _create(std::optional<std::string>& request_id, ChatCompletionRequest request); // class ChatCompletion -> create()
  std::string _response_to_str(ChatCompletionResponse& response);

  // Functions in engine_base.py
  std::optional<ChatCompletionStreamResponse> process_chat_completion_stream_output(std::vector<CallbackStreamOutput>& delta_outputs, mlc::llm::json_ffi::ChatCompletionRequest& request, Optional<String> request_id, bool use_function_calling, Array<Optional<String>> finish_reasons);
  ChatCompletionResponse wrap_chat_completion_response(std::string& request_id, std::string& model, std::vector<std::string>& output_texts, std::vector<std::string>& finish_reasons);  
private:
  std::string _model;
  Conversation _conv_template;
  std::vector<mlc::llm::json_ffi::ModelConfig> _model_config_list;
  mlc::llm::Tokenizer _tokenizer;
  tvm::runtime::Module _engine_module;
  std::thread _background_loop_thread;
  std::thread _background_stream_back_loop_thread;
  bool _terminated;
  mlc::llm::serve::EngineConfig _engine_config;
  int _max_input_sequence_length;
  std::optional<mlc::llm::serve::EventTraceRecorder> _trace_recorder;
  BlockingQueue<tvm::ffi::Array<mlc::llm::serve::RequestStreamOutput>> _sync_output_queue;
  std::vector<mlc::llm::TextStreamer> _sync_text_streamers;
  bool _is_end;
  ChatCompletionRequest _request;
  Optional<String> _request_id;
};


#endif // MLC_SERVE_LLM_SEGMENT_RUNNER_SEGMENT_RUNNER_H_