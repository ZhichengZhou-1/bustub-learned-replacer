//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// learned_replacer.h
//
// Identification: src/include/buffer/learned_replacer.h
//
//===----------------------------------------------------------------------===//

#pragma once

#include <fstream>
#include <list>
#include <mutex>  // NOLINT
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "buffer/arc_replacer.h"  // for AccessType
#include "common/config.h"
#include "common/macros.h"

#include "onnxruntime_cxx_api.h"
namespace bustub {

/**
 * FrameMeta holds the per-frame features we track for the learned model.
 * These become the input features for the ONNX model at eviction time.
 */
struct FrameMeta {
  page_id_t page_id_{-1};

  // Feature 1: how many times this frame has been accessed
  float frequency_{0};

  // Feature 2: timestamp of the most recent access (higher = more recent)
  float recency_{0};

  // Feature 3: average time between accesses (lower = accessed more regularly)
  float avg_interval_{0};

  // Feature 4: encoded access type of the most recent access
  // 0=Unknown, 1=Lookup, 2=Scan, 3=Index
  float access_type_{0};

  // Internal tracking (not passed to model directly)
  size_t last_access_timestamp_{0};
  size_t total_interval_{0};
  bool is_evictable_{false};
};

/**
 * LearnedReplacer uses a trained ONNX model to predict next-access time
 * and evicts the frame least likely to be accessed soon.
 *
 * Drop-in replacement for ArcReplacer — same public interface.
 */
class LearnedReplacer {
 public:
  explicit LearnedReplacer(size_t num_frames, const std::string &model_path = "learned_replacer.onnx");

  DISALLOW_COPY_AND_MOVE(LearnedReplacer);

  ~LearnedReplacer();

  auto Evict() -> std::optional<frame_id_t>;

  void RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType access_type = AccessType::Unknown);

  void SetEvictable(frame_id_t frame_id, bool set_evictable);

  void Remove(frame_id_t frame_id);

  auto Size() -> size_t;

 private:
  /**
   * Run a forward pass through the ONNX model for a single frame.
   * Returns a score — higher score means more likely to be accessed soon
   * (i.e. do NOT evict). Lower score = safer to evict.
   */
  auto RunInference(const FrameMeta &meta) -> float;

  /**
   * Fallback eviction when ONNX model is not loaded.
   * Uses LRU (least recently accessed frame).
   */
  auto FallbackEvict() -> std::optional<frame_id_t>;

  // Per-frame metadata used as model features
  std::unordered_map<frame_id_t, FrameMeta> frame_meta_;

  // Number of evictable frames (replacer size)
  size_t curr_size_{0};

  // Max frames this replacer tracks
  size_t replacer_size_;

  // Monotonically increasing timestamp for recency tracking
  size_t current_timestamp_{0};

  // Thread safety
  std::mutex latch_;

  // ONNX Runtime
  Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "learned_replacer"};
  std::unique_ptr<Ort::Session> ort_session_{nullptr};
  Ort::SessionOptions ort_session_options_;

  // Trace collection
  std::ofstream trace_file_;
  bool tracing_enabled_{false};

  bool model_loaded_{false};
};

}  // namespace bustub