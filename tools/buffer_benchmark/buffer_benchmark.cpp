#include <iostream>
#include <random>
#include <vector>
#include "buffer/learned_replacer.h"

using namespace bustub;  // NOLINT

const size_t POOL_SIZE = 64;
const size_t NUM_PAGES = 200;
const size_t OPS = 10000;

void RunWorkload(const std::string &name, std::vector<page_id_t> &sequence, float workload_feat) {
  std::cout << "Running workload: " << name << "\n";
  LearnedReplacer replacer(POOL_SIZE, "learned_replacer.onnx", name + "_trace.csv");

  size_t hits = 0;
  size_t evictions = 0;

  // Simple in-memory page table: page_id -> frame_id
  std::unordered_map<page_id_t, frame_id_t> page_table;
  std::unordered_map<frame_id_t, page_id_t> frame_table;
  std::vector<frame_id_t> free_frames;
  for (frame_id_t i = 0; i < static_cast<frame_id_t>(POOL_SIZE); i++) {
    free_frames.push_back(i);
  }

  for (page_id_t page_id : sequence) {
    // Check if page is already in pool
    if (page_table.count(page_id) > 0) {
      frame_id_t fid = page_table[page_id];
      replacer.RecordAccess(fid, page_id, AccessType::Lookup, workload_feat);
      hits++;
      continue;
    }

    // Need to bring page in
    frame_id_t fid;
    if (!free_frames.empty()) {
      fid = free_frames.back();
      free_frames.pop_back();
    } else {
      // Evict
      auto victim = replacer.Evict();
      if (!victim.has_value()) {
        std::cerr << "No frame to evict!\n";
        continue;
      }
      fid = victim.value();
      page_id_t evicted_page = frame_table[fid];
      page_table.erase(evicted_page);
      evictions++;
    }

    // Load page into frame
    page_table[page_id] = fid;
    frame_table[fid] = page_id;
    replacer.RecordAccess(fid, page_id, AccessType::Lookup, workload_feat);
    replacer.SetEvictable(fid, true);
  }

  double hit_rate = static_cast<double>(hits) / sequence.size() * 100.0;
  std::cout << "  Hit rate:   " << hit_rate << "%\n";
  std::cout << "  Hits:       " << hits << "\n";
  std::cout << "  Evictions:  " << evictions << "\n\n";
}

int main() {
  std::mt19937 rng(42);

  // Workload 1: Sequential scan (pages 0..NUM_PAGES in order, repeated)
  std::vector<page_id_t> sequential;
  for (size_t i = 0; i < OPS; i++) {
    sequential.push_back(static_cast<page_id_t>(i % NUM_PAGES));
  }

  // Workload 2: Random access (uniform random over all pages)
  std::uniform_int_distribution<page_id_t> dist(0, NUM_PAGES - 1);
  std::vector<page_id_t> random_access;
  for (size_t i = 0; i < OPS; i++) {
    random_access.push_back(dist(rng));
  }

  // Workload 3: Mixed (80% hot set of 20 pages, 20% random)
  std::uniform_int_distribution<page_id_t> hot_dist(0, 19);
  std::uniform_int_distribution<page_id_t> cold_dist(20, NUM_PAGES - 1);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  std::vector<page_id_t> mixed;
  for (size_t i = 0; i < OPS; i++) {
    mixed.push_back(coin(rng) < 0.8 ? hot_dist(rng) : cold_dist(rng));
  }

  RunWorkload("sequential", sequential, 0.0f);
  RunWorkload("random", random_access, 1.0f);
  RunWorkload("mixed", mixed, 2.0f);

  return 0;
}