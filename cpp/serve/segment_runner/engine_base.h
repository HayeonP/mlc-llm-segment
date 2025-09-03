#ifndef MLC_LLM_FRONTEND_ENGINE_BASE_H
#define MLC_LLM_FRONTEND_ENGINE_BASE_H

#include <string>
#include <vector>

struct ModelInfo{
    std::string model = "";
    std::string model_lib = "";
};

using ModelInfo = struct ModelInfo;

#endif