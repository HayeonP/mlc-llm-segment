#ifndef MLC_LLM_SERVE_SEGMENT_RUNNER_DATA_TYPES_H_
#define MLC_LLM_SERVE_SEGMENT_RUNNER_DATA_TYPES_H_

#include <iostream>

#include <tvm/ffi/string.h>
#include <tvm/ffi/optional.h>
#include <tvm/runtime/int_tuple.h>
#include <tvm/ffi/container/array.h>

using namespace tvm;
using namespace ffi;
using TokenIds = IntTuple; // tvm::ffi::Shape
using String = tvm::ffi::String;

struct CallbackStreamOutput {  
  String delta_text;
  Optional<Array<String>> delta_logprob_json_strs;
  Optional<String> finish_reason;
  Optional<String> request_final_usage_json_str;
};

struct SingleRequestStreamOutput {
  IntTuple delta_token_ids;
  Optional<Array<String>> delta_logprob_json_strs;
  Optional<String> finish_reason;
  Optional<String> request_final_usage_json_str;
  String extra_prefix_string;
};

struct TopLogProbs{
  std::string token;
  float logprob;
  std::optional<std::vector<int>> bytes;
};

using TopLogProbs = struct TopLogProbs;

struct LogProbsContent{
  std::string str;
  float logrpob;
  std::optional<std::vector<int>> bytes;
  std::vector<TopLogProbs> top_logprobs;
};

using LogProbsContent = struct LogProbsContent;

struct LogProbs{
  std::vector<LogProbsContent> content;
};

using LogProbs = struct LogProbs;

using CallbackStreamOutput = struct CallbackStreamOutput;
using SingleRequestStreamOutput = struct SingleRequestStreamOutput;

#endif // MLC_LLM_SERVE_SEGMENT_RUNNER_DATA_TYPES_H_