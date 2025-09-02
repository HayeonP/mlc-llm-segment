#ifndef MLC_LLM_SERVE_SEGMENT_RUNNER_UTILS_H_
#define MLC_LLM_SERVE_SEGMENT_RUNNER_UTILS_H_

#include <string>
#include <fstream>
#include <picojson.h>

#include "../../json_ffi/conv_template.h"
#include "../../json_ffi/openai_api_protocol.h"

#include <tvm/runtime/int_tuple.h>

using Conversation = mlc::llm::json_ffi::Conversation;
using ChatCompletionMessageContent = mlc::llm::json_ffi::ChatCompletionMessageContent;
using ChatCompletionMessage = mlc::llm::json_ffi::ChatCompletionMessage;
using ChatCompletionRequest = mlc::llm::json_ffi::ChatCompletionRequest;
using TokenIds = IntTuple;

namespace mlc{
namespace llm{
namespace utils{

std::string ReadJSONAsString(const std::string& json_path);

bool HasKey(const picojson::object& o, const char* k);

const picojson::value* GetPtr(const picojson::object& o, const char* k);

void ExpectType(bool cond, const char* msg);

mlc::llm::serve::EngineConfig ParseEngineConfigFromJSONString(const std::string& json_str);

std::string Uuid4Hex();

void PrintRequest(mlc::llm::json_ffi::ChatCompletionRequest rq);

void PrintConversation(mlc::llm::json_ffi::Conversation conv);

std::string ReplaceString(std::string s, const std::string& from, const std::string& to);

std::string GetRolePlaceholder(const std::string& role);

std::string ToUpper(const std::string& s);

// TODO: This function assumes messages are string only
std::vector<std::string> _combine_consecutive_messages(std::vector<std::string> messages);

std::vector<std::string> ConvertConversationToPrompt(Conversation& conv);

static std::vector<int> TokenIds2IntVec(const TokenIds& t);

static TokenIds IntVec2TokenIds(std::vector<int> v);

IntTuple AppendIntTuple(const IntTuple& a, const IntTuple& b);

void PrintIntTuple(const IntTuple& t);

void IntTupleToInt64Vector(const IntTuple& int_tuple, std::vector<int64_t>& int_vector);

void IntTupleToInt32Vector(const IntTuple& int_tuple, std::vector<int32_t>& int_vector);

void StringArrayToStdArryVector(const tvm::ffi::Array<tvm::ffi::String>& string_array, std::vector<std::string>& std_string_vector);

std::string FinishReasonToStr(mlc::llm::json_ffi::FinishReason& finish_reason);

mlc::llm::json_ffi::FinishReason StrToFinishReason(std::string& finish_reason);

} // using namespace mlc
} // using namespace llm
} // using namespace utils

#endif // MLC_LLM_SERVE_SEGMENT_RUNNER_UTILS_H_