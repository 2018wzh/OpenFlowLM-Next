#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <array>
#include "nlohmann/json.hpp"

namespace open_qwen36 {
class Core;
void require_capture_layout(const nlohmann::json& layout);

// Synchronous, bounded-buffer writer: device execution never outruns disk writes.
class CaptureWriter {
public:
    explicit CaptureWriter(const std::string& directory, bool live = false);
    nlohmann::json tensor(const std::string& name, const std::string& dtype,
                         nlohmann::json shape, const void* data, size_t bytes);
    void event(nlohmann::json value);
    void finish();
private:
    std::ofstream data_, events_;
    uint64_t offset_ = 0, sequence_ = 0;
    bool live_ = false;
};

class MoeCapture {
public:
    MoeCapture(CaptureWriter& writer, const nlohmann::json& layout);
    void begin_case(const nlohmann::json& value);
    void begin_run(const std::string& name, int ablate_rank = -1);
    void end_run();
    void input(const std::string& phase, int position, int token);
    int route(Core& core, int layer, const uint8_t* record);
    void after_layer(Core& core, int layer);
    void logits(Core& core, int next_token = -1);
    int target(int rank) const;
private:
    nlohmann::json context(int layer = -1) const;
    bool selected(int layer) const;
    CaptureWriter& writer_;
    nlohmann::json layout_, case_;
    std::string run_, phase_;
    int position_ = 0, token_ = 0, ablate_rank_ = -1;
    int references_[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    int current_ids_[8] = {};
    struct Load {
        std::array<uint64_t,256> counts{};
        std::array<double,256> weights{};
        uint64_t positions = 0;
        double entropy = 0;
    };
    std::array<Load,40> loads_{};
    nlohmann::json phases_;
};

int run_capture(Core& core, const std::string& config, const std::string& directory);
}
