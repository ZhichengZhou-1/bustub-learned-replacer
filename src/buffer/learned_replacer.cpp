//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// learned_replacer.cpp
//
// Identification: src/buffer/learned_replacer.cpp
//
//===----------------------------------------------------------------------===//

#include "buffer/learned_replacer.h"
#include <algorithm>
#include <stdexcept>
#include "common/exception.h"

namespace bustub {

LearnedReplacer::LearnedReplacer(size_t num_frames, const std::string &model_path, const std::string &trace_path)
    : replacer_size_(num_frames) {
  // Load ONNX model
  try {
    ort_session_ = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), ort_session_options_);
    model_loaded_ = true;
    std::cout << "[LearnedReplacer] Model loaded from " << model_path << "\n";
  } catch (const Ort::Exception &e) {
    std::cout << "[LearnedReplacer] Model not found, using LRU fallback: " << e.what() << "\n";
    model_loaded_ = false;
  }

  // Open trace file for writing
  trace_file_.open(trace_path, std::ios::out | std::ios::trunc);

  if (trace_file_.is_open()) {
    trace_file_ << "timestamp,frame_id,page_id,access_type\n";
    tracing_enabled_ = true;
    std::cout << "[LearnedReplacer] Trace writing to: " << trace_path << "\n";
  } else {
    std::cout << "[LearnedReplacer] FAILED to open trace file: " << trace_path << "\n";
  }
}

LearnedReplacer::~LearnedReplacer() {
  // TODO(Phase 5): free ONNX session and env
  if (trace_file_.is_open()) {
    trace_file_.close();
  }
}

auto LearnedReplacer::Evict() -> std::optional<frame_id_t> {
  std::scoped_lock lock(latch_);

  if (curr_size_ == 0) {
    return std::nullopt;
  }

  if (model_loaded_) {
    // Score every evictable frame and evict the one with lowest score
    frame_id_t best_frame = -1;
    float lowest_score = std::numeric_limits<float>::max();

    for (auto &[fid, meta] : frame_meta_) {
      if (!meta.is_evictable_) {
        continue;
      }
      float score = RunInference(meta);
      if (score < lowest_score) {
        lowest_score = score;
        best_frame = fid;
      }
    }

    if (best_frame == -1) {
      return std::nullopt;
    }

    frame_meta_.erase(best_frame);
    curr_size_--;
    return best_frame;
  }

  // Fallback: LRU eviction
  return FallbackEvict();
}

void LearnedReplacer::RecordAccess(frame_id_t frame_id, page_id_t page_id, AccessType access_type) {
  std::scoped_lock lock(latch_);

  if (static_cast<size_t>(frame_id) >= replacer_size_) {
    throw Exception("LearnedReplacer: invalid frame_id in RecordAccess");

    // Write to trace file
    if (tracing_enabled_) {
      trace_file_ << current_timestamp_ << "," << frame_id << "," << page_id << "," << static_cast<int>(access_type)
                  << "\n";
      trace_file_.flush();
    }
  }

  current_timestamp_++;

  auto &meta = frame_meta_[frame_id];
  meta.page_id_ = page_id;

  // Update interval tracking before updating last timestamp
  if (meta.frequency_ > 0) {
    size_t interval = current_timestamp_ - meta.last_access_timestamp_;
    meta.total_interval_ += interval;
    meta.avg_interval_ = static_cast<float>(meta.total_interval_) / meta.frequency_;
  }

  // Update features
  meta.time_since_last_access_ =
      (meta.frequency_ > 0) ? static_cast<float>(current_timestamp_ - meta.last_access_timestamp_) : 999.0f;
  meta.frequency_++;
  meta.last_access_timestamp_ = current_timestamp_;
  meta.access_type_ = static_cast<float>(access_type);

  // Write to trace file
  if (tracing_enabled_) {
    trace_file_ << current_timestamp_ << "," << frame_id << "," << page_id << "," << static_cast<int>(access_type)
                << "\n";
    trace_file_.flush();
  }
}

void LearnedReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  std::scoped_lock lock(latch_);

  if (static_cast<size_t>(frame_id) >= replacer_size_) {
    throw Exception("LearnedReplacer: invalid frame_id in SetEvictable");
  }

  auto it = frame_meta_.find(frame_id);
  if (it == frame_meta_.end()) {
    return;
  }

  auto &meta = it->second;
  if (meta.is_evictable_ && !set_evictable) {
    curr_size_--;
  } else if (!meta.is_evictable_ && set_evictable) {
    curr_size_++;
  }
  meta.is_evictable_ = set_evictable;
}

void LearnedReplacer::Remove(frame_id_t frame_id) {
  std::scoped_lock lock(latch_);

  auto it = frame_meta_.find(frame_id);
  if (it == frame_meta_.end()) {
    return;
  }

  if (!it->second.is_evictable_) {
    throw Exception("LearnedReplacer: Remove called on non-evictable frame");
  }

  frame_meta_.erase(it);
  curr_size_--;
}

auto LearnedReplacer::Size() -> size_t {
  std::scoped_lock lock(latch_);
  return curr_size_;
}

auto LearnedReplacer::RunInference(const FrameMeta &meta) -> float {
  if (!model_loaded_ || ort_session_ == nullptr) {
    return 0.0f;
  }

  try {
    // Scaler params from python/scaler_params.json
    // features: [frequency, time_since_last_access, avg_interval, access_type]
    constexpr std::array<float, 4> MEANS = {1163.4129f, 7.7876f, 6.7221f, 1.0f};
    constexpr std::array<float, 4> STDS = {1362.5994f, 29.1223f, 3.4646f, 1.0f};

    std::array<float, 4> raw = {meta.frequency_, meta.time_since_last_access_, meta.avg_interval_, meta.access_type_};

    // Apply normalization
    std::array<float, 4> features{};
    for (int i = 0; i < 4; i++) {
      features[i] = (raw[i] - MEANS[i]) / STDS[i];
    }

    std::array<int64_t, 2> input_shape{1, 4};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, features.data(), 4, input_shape.data(), 2);

    const char *input_names[] = {"features"};
    const char *output_names[] = {"score"};

    auto output = ort_session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float score = *output[0].GetTensorData<float>();

    // DEBUG: print first 10 eviction decisions
    static int call_count = 0;
    call_count++;
    if (call_count % 1000 == 0) {
      std::cout << "[DEBUG] freq=" << meta.frequency_ << " tsla=" << meta.time_since_last_access_
                << " avg_int=" << meta.avg_interval_ << " score=" << score << "\n";
    }

    return score;

  } catch (const Ort::Exception &e) {
    return 0.0f;
  }
}

auto LearnedReplacer::FallbackEvict() -> std::optional<frame_id_t> {
  // LRU: evict the frame with the smallest recency timestamp
  frame_id_t lru_frame = -1;
  size_t oldest_timestamp = std::numeric_limits<size_t>::max();

  for (auto &[fid, meta] : frame_meta_) {
    if (!meta.is_evictable_) {
      continue;
    }
    if (meta.last_access_timestamp_ < oldest_timestamp) {
      oldest_timestamp = meta.last_access_timestamp_;
      lru_frame = fid;
    }
  }

  if (lru_frame == -1) {
    return std::nullopt;
  }

  frame_meta_.erase(lru_frame);
  curr_size_--;
  return lru_frame;
}

}  // namespace bustub