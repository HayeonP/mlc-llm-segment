/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/segment_runner_engine.h
 * \brief The header of segment runner serving engine in MLC LLM.
 */
#ifndef MLC_LLM_SERVE_SEGMENT_RUNNER_SEGMENT_RUNNER_ENGINE_H_
#define MLC_LLM_SERVE_SEGMENT_RUNNER_SEGMENT_RUNNER_ENGINE_H_

#include <picojson.h>

#include "../../serve/data.h"
#include "../../serve/engine.h"
#include "../../serve/request.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief The interface segment runner engine in MLC LLM.
 * The segment runner engine keeps running a background request processing
 * loop on a standalone thread. Ensuring thread safety, it exposes
 * `AddRequest` and `AbortRequest` to receive new requests or
 * abortions from other threads, and the internal request processing
 * is backed by a normal engine wrapped inside.
 */
class SegmentRunnerEngine {

 public:
  /*! \brief Create a SegmentRunnerEngine. */
  // static std::unique_ptr<SegmentRunnerEngine> Create();
  static std::unique_ptr<SegmentRunnerEngine> Create();


  virtual ~SegmentRunnerEngine() = default;

  /*!
   * \brief Initialize the threaded engine from packed arguments in PackedArgs.
   * \param device The device where to run models.
   * \param request_stream_callback The request stream callback function to.
   * \param trace_recorder Event trace recorder for requests.
   */
  virtual void InitSegmentRunnerEngine(Device device, Optional<Function> request_stream_callback,
                                  Optional<EventTraceRecorder> trace_recorder) = 0;

  /*!
   * \brief Reload the engine with the new engine config.
   * \param engine_config_json_str The engine config JSON string.
   */
  virtual void Reload(String engine_config_json_str) = 0;

  /*! \brief Unload the background engine. */
  virtual void Unload() = 0;

  /*! \brief Reset the engine to the initial state. */
  virtual void Reset() = 0;

  /*!
   * \brief Notify the SegmentRunnerEngine to exit the background
   * request processing loop. This method is invoked by threads
   * other than the engine-driving thread.
   */
  virtual void ExitBackgroundLoop() = 0;

  /*! \brief Add a new request to the engine. */
  virtual void AddRequest(Request request) = 0;

  /*! \brief Abort the input request (specified by id string) from engine. */
  virtual void AbortRequest(const String& request_id) = 0;

  /*! \brief Add a new request to the engine. */
  virtual void RunSegment() = 0;

  /************** Query/Profile/Debug **************/

  /*! \brief Return the default generation config. */
  virtual GenerationConfig GetDefaultGenerationConfig() const = 0;

  /*! \brief Return the complete engine config. */
  virtual EngineConfig GetCompleteEngineConfig() const = 0;

  /*! \brief Call the given global function on all workers. Only for debug purpose. */
  virtual void DebugCallFuncOnAllAllWorker(const String& func_name, Optional<String> func_args) = 0;




};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SEGMENT_RUNNER_SEGMENT_RUNNER_ENGINE_H_
