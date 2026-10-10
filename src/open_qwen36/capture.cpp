#include "open_qwen36/capture.hpp"
#include "open_qwen36/core.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

namespace open_qwen36 {
using nlohmann::json;

void require_capture_layout(const json& layout) {
    if (layout.is_null() || layout.value("version", 0) != 1)
        throw std::runtime_error("MoE capture requires capture-capable kernels (export --capture)");
    const auto capabilities = layout.value("capabilities", std::vector<std::string>{});
    for (const char* name : {"native_router", "native_moe_vectors", "native_expert_intermediates",
                             "native_bf16_down_input", "masked_contribution"})
        if (std::find(capabilities.begin(), capabilities.end(), name) == capabilities.end())
            throw std::runtime_error(std::string("capture kernel missing capability: ") + name);
    const size_t size = layout.at("bytes").get<size_t>();
    for (const char* name : {"up", "gate", "silu", "product", "down", "accumulator", "down_input", "shared_gate"}) {
        const auto& field = layout.at("fields").at(name);
        const size_t count = std::string(name) == "shared_gate" ? 8 : 9;
        if (field.at("offset").get<size_t>() + count * field.at("stride").get<size_t>() > size)
            throw std::runtime_error(std::string("capture field exceeds buffer: ") + name);
    }
    for (const char* name : {"linear_attention", "full_attention"})
        for (const char* field : {"base", "xm", "residual"})
            (void)layout.at("layer_types").at(name).at(field).get<size_t>();
}

CaptureWriter::CaptureWriter(const std::string& directory, bool live) : live_(live) {
    const std::filesystem::path d(directory);
    std::filesystem::create_directories(d);
    if (std::filesystem::exists(d / "events.jsonl") || std::filesystem::exists(d / "tensors.bin"))
        throw std::runtime_error("capture output already exists; choose a new directory");
    data_.open(d / "tensors.bin", std::ios::binary);
    events_.open(d / "events.jsonl", std::ios::binary);
    data_.exceptions(std::ios::badbit | std::ios::failbit);
    events_.exceptions(std::ios::badbit | std::ios::failbit);
}

json CaptureWriter::tensor(const std::string& name, const std::string& dtype,
                           json shape, const void* data, size_t bytes) {
    json t = {{"name", name}, {"dtype", dtype}, {"shape", std::move(shape)},
              {"offset", offset_}, {"length", bytes}, {"source", "native_openflowlm"}};
    data_.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    offset_ += bytes;
    return t;
}

void CaptureWriter::event(json value) {
    value["event_seq"] = sequence_++;
    // Publish tensor bytes before their index becomes visible to live readers.
    if (live_) data_.flush();
    events_ << value.dump() << '\n';
    if (live_) events_.flush();
}

void CaptureWriter::finish() {
    // Completion is published only after every tensor has reached the file.
    data_.flush();
    event({{"kind", "complete"}, {"tensor_bytes", offset_}});
    events_.flush();
}

MoeCapture::MoeCapture(CaptureWriter& writer, const json& layout)
    : writer_(writer), layout_(layout) {
    require_capture_layout(layout_);
}

void MoeCapture::begin_case(const json& value) {
    end_run();
    case_ = value;
    std::fill(std::begin(references_), std::end(references_), -1);
    writer_.event({{"kind", "case"}, {"case", case_.at("id")}, {"text", case_.at("text")},
                   {"input_ids", case_.at("ids")}, {"tokens", case_.at("tokens")},
                   {"capture_positions", case_.at("capture_positions")},
                   {"vector_layers", case_.at("vector_layers")},
                   {"micro_layers", case_.at("micro_layers")},
                   {"micro_rank", case_.value("micro_rank", 0)},
                   {"target_position", case_.at("target_position")},
                   {"ablate_layer", case_.at("ablate_layer")}});
}

void MoeCapture::begin_run(const std::string& name, int ablate_rank) {
    end_run();
    run_ = name;
    loads_ = {};
    phases_ = json::object();
    ablate_rank_ = ablate_rank;
    writer_.event({{"kind", "run"}, {"case", case_.at("id")}, {"run", run_},
                   {"ablate_rank", ablate_rank}, {"ablate_expert", ablate_rank < 0 ? -1 : target(ablate_rank)}});
}

void MoeCapture::end_run() {
    if (run_.empty()) return;
    json layers = json::array();
    for (size_t l = 0; l < loads_.size(); ++l) {
        const auto& a = loads_[l];
        layers.push_back({{"layer",l},{"positions",a.positions},{"expert_dispatch_counts",a.counts},
                          {"expert_weight_sums",a.weights},{"mean_router_entropy_nats",a.positions ? a.entropy/a.positions : 0.0}});
    }
    writer_.event({{"kind","run_summary"},{"case",case_.at("id")},{"run",run_},{"source","derived"},
                   {"phases",phases_},{"layers",layers},{"load_definition","original Top8 dispatches and weights, including masked slot"}});
    run_.clear();
}

void MoeCapture::input(const std::string& phase, int position, int token) {
    phase_ = phase; position_ = position; token_ = token;
}

json MoeCapture::context(int layer) const {
    return {{"case", case_.at("id")}, {"run", run_}, {"phase", phase_},
            {"position", position_}, {"input_token_id", token_}, {"layer", layer}};
}

bool MoeCapture::selected(int layer) const {
    const auto positions = case_.at("capture_positions").get<std::vector<int>>();
    const auto layers = case_.at("vector_layers").get<std::vector<int>>();
    return std::find(positions.begin(), positions.end(), position_) != positions.end() &&
           std::find(layers.begin(), layers.end(), layer) != layers.end();
}

int MoeCapture::target(int rank) const {
    if (rank < 0 || rank >= 8 || references_[rank] < 0)
        throw std::runtime_error("baseline did not resolve the requested ablation expert");
    return references_[rank];
}

int MoeCapture::route(Core& core, int layer, const uint8_t* record) {
    std::memcpy(current_ids_, record + 1024, 32);
    const bool at_target = position_ == case_.at("target_position").get<int>() &&
                           layer == case_.at("ablate_layer").get<int>();
    if (at_target && run_ == "baseline_generate")
        std::copy(std::begin(current_ids_), std::end(current_ids_), std::begin(references_));
    int slot = -1;
    if (at_target && ablate_rank_ >= 0) {
        const int expert = target(ablate_rank_);
        for (int i = 0; i < 8; ++i) if (current_ids_[i] == expert) slot = i;
        if (slot < 0) throw std::runtime_error("ablation target absent at the intervention point");
    }
    json e = context(layer);
    e["kind"] = "route";
    e["ablate_slot"] = slot;
    e["tensors"] = json::array({
        writer_.tensor("router_logits", "f32", {256}, record + layout_.at("router_logits_offset").get<size_t>(), 1024),
        writer_.tensor("router_probabilities", "f32", {256}, record, 1024),
        writer_.tensor("expert_ids", "i32", {8}, record + 1024, 32),
        writer_.tensor("expert_weights", "f32", {8}, record + 1056, 32)});
    float probabilities[256], weights[8];
    std::memcpy(probabilities, record, sizeof(probabilities));
    std::memcpy(weights, record + 1056, sizeof(weights));
    auto& load = loads_.at(layer);
    ++load.positions;
    for (float p : probabilities) if (p > 0) load.entropy -= static_cast<double>(p)*std::log(static_cast<double>(p));
    for (int i=0;i<8;++i) { ++load.counts.at(current_ids_[i]); load.weights.at(current_ids_[i]) += weights[i]; }
    if (layer==0) phases_[phase_] = phases_.value(phase_, uint64_t(0)) + 1;
    writer_.event(std::move(e));
    if (selected(layer)) {
        const auto& lt = layout_.at("layer_types").at(core.manifest().layer_type(layer).name);
        std::vector<uint8_t> xm(2048 * 2), residual(2048 * 4);
        core.read_act(layer, lt.at("xm").get<size_t>(), xm.size(), xm.data());
        core.read_act(layer, lt.at("residual").get<size_t>(), residual.size(), residual.data());
        e = context(layer); e["kind"] = "moe_input";
        e["tensors"] = json::array({writer_.tensor("normalized_input", "bf16", {2048}, xm.data(), xm.size()),
                                    writer_.tensor("residual_before", "f32", {2048}, residual.data(), residual.size())});
        writer_.event(std::move(e));
    }
    return slot;
}

void MoeCapture::after_layer(Core& core, int layer) {
    const auto& lt = layout_.at("layer_types").at(core.manifest().layer_type(layer).name);
    const auto& gate_field = layout_.at("fields").at("shared_gate");
    float gate;
    core.read_act(layer, lt.at("base").get<size_t>() + gate_field.at("offset").get<size_t>(), sizeof(gate), reinterpret_cast<uint8_t*>(&gate));
    json gate_event = context(layer);
    gate_event.update({{"kind","shared_gate"},{"value",gate},{"source","native_openflowlm"}});
    gate_event["tensors"] = json::array({writer_.tensor("shared_gate","f32",{1},&gate,sizeof(gate))});
    writer_.event(std::move(gate_event));
    if (!selected(layer)) return;
    std::vector<uint8_t> bytes(layout_.at("bytes").get<size_t>()), residual(2048 * 4);
    core.read_act(layer, lt.at("base").get<size_t>(), bytes.size(), bytes.data());
    core.read_residual(residual.data());
    auto tensor = [&](const std::string& field, const std::string& name, int slot, size_t n, json shape) {
        const auto& f = layout_.at("fields").at(field);
        const size_t off = f.at("offset").get<size_t>() + slot * f.at("stride").get<size_t>();
        if (off + n > bytes.size()) throw std::runtime_error("capture manifest tensor outside its buffer");
        return writer_.tensor(name, f.at("dtype"), shape, bytes.data() + off, n);
    };
    json e = context(layer); e["kind"] = "moe_output"; e["expert_ids"] = current_ids_;
    e["tensors"] = json::array();
    for (int s = 0; s < 9; ++s) {
        json t = tensor("down", s == 8 ? "shared_output" : "expert_output", s, 2048 * 4, {2048});
        t["slot"] = s;
        if (s < 8) t["expert_id"] = current_ids_[s];
        e["tensors"].push_back(std::move(t));
    }
    e["tensors"].push_back(tensor("accumulator", "routed_mixture", 7, 2048 * 4, {2048}));
    e["tensors"].push_back(tensor("shared_gate", "shared_gate", 0, 4, {1}));
    e["tensors"].push_back(writer_.tensor("residual_after", "f32", {2048}, residual.data(), residual.size()));
    writer_.event(std::move(e));
    const auto micro_layers = case_.at("micro_layers").get<std::vector<int>>();
    if (std::find(micro_layers.begin(), micro_layers.end(), layer) == micro_layers.end()) return;
    int expert = current_ids_[case_.value("micro_rank", 0)];
    // Freeze the microscope's identity to the baseline when it is the intervention point.
    if (position_ == case_.at("target_position").get<int>() && layer == case_.at("ablate_layer").get<int>())
        expert = target(case_.value("micro_rank", 0));
    int slot = -1;
    for (int s = 0; s < 8; ++s) if (current_ids_[s] == expert) slot = s;
    e = context(layer); e["kind"] = "expert_detail"; e["expert_id"] = expert; e["slot"] = slot;
    e["tensors"] = json::array();
    if (slot < 0) e["missing"] = "baseline expert is not routed at this position";
    else {
        for (const char* f : {"gate", "up", "silu", "product"})
            e["tensors"].push_back(tensor(f, f, slot, 512 * 4, {512}));
        // Every core prepares the entire hidden. Core 0's rounded values are the
        // exact vector consumed by its down table, not a host reconstruction.
        e["tensors"].push_back(tensor("down_input", "down_input", slot, 512 * 2, {512}));
        e["tensors"].push_back(tensor("down", "down", slot, 2048 * 4, {2048}));
    }
    writer_.event(std::move(e));
}

void MoeCapture::logits(Core& core, int next_token) {
    json e = context(); e["kind"] = "logits";
    if (next_token >= 0) e["next_token_id"] = next_token;
    e["tensors"] = json::array({writer_.tensor("logits", "f32", {core.real_vocab()},
                                               core.logits().data(), core.real_vocab() * 4)});
    writer_.event(std::move(e));
}

static double log_probability(Core& core, int id) {
    const auto& v = core.logits();
    if (id < 0 || static_cast<size_t>(id) >= core.real_vocab())
        throw std::runtime_error("teacher-forced token outside real vocabulary");
    const double m = *std::max_element(v.begin(), v.begin() + core.real_vocab());
    double sum = 0;
    for (size_t i = 0; i < core.real_vocab(); ++i) sum += std::exp(static_cast<double>(v[i]) - m);
    return static_cast<double>(v[id]) - m - std::log(sum);
}

int run_capture(Core& core, const std::string& config, const std::string& directory) {
    std::ifstream f(config);
    json c; f >> c;
    if (c.at("version").get<int>() != 1 || c.at("cases").empty())
        throw std::runtime_error("capture config needs version 1 and nonempty cases");
    if (core.num_layers() != 40 || core.manifest().hidden != 2048 || core.manifest().family != "qwen36moe")
        throw std::runtime_error("capture requires the complete Qwen3.6-35B-A3B model");
    const auto& layout = core.manifest().moe_capture;
    if (layout.is_null()) throw std::runtime_error("kernel set has no MoE capture capability");
    CaptureWriter writer(directory, c.value("live", false));
    writer.event({{"kind", "metadata"}, {"version", 1}, {"backend", "native_openflowlm"},
                  {"model", "Qwen3.6-35B-A3B"}, {"layers", 40}, {"experts", 256}, {"shared_experts",1}, {"topk", 8},
                  {"byte_order", "little"}, {"precision", "native Q4NX packing; BF16 activations; F32 accumulators and logits"},
                  {"spec", core.manifest().hf_config_check}, {"capture_layout_version", 1},
                  {"execution_quantization", core.manifest().model_spec.value("quant", json("q4_1"))},
                  {"moe_precision", {{"routed_weights", "Q4_1"}, {"shared_weights", "Q4_1"},
                                     {"down_input", "BF16"}, {"accumulation", "split-BF16 approximate F32 MAC"}}},
                  {"schedule", "per_token_npu"}});
    MoeCapture capture(writer, layout);
    core.set_capture(&capture);
    try {
        auto check_cancel = [&]() {
            if (std::filesystem::exists(std::filesystem::path(directory) / "cancel"))
                throw std::runtime_error("capture cancelled");
        };
        for (const auto& value : c.at("cases")) {
            const auto ids = value.at("ids").get<std::vector<int>>();
            const int max_tokens = value.value("max_tokens", 8);
            if (ids.empty() || max_tokens < 1 || value.at("target_position").get<size_t>() >= ids.size())
                throw std::runtime_error("invalid capture prompt or generation length");
            for (const char* name : {"ablate_layer"})
                if (value.at(name).get<int>() < 0 || value.at(name).get<int>() >= 40)
                    throw std::runtime_error("capture layer outside model");
            for (int l : value.at("vector_layers").get<std::vector<int>>())
                if (l < 0 || l >= 40) throw std::runtime_error("capture vector layer outside model");
            for (int l : value.at("micro_layers").get<std::vector<int>>())
                if (l < 0 || l >= 40) throw std::runtime_error("capture microscope layer outside model");
            const int micro_rank = value.value("micro_rank", 0);
            if (micro_rank < 0 || micro_rank >= 8)
                throw std::runtime_error("capture micro rank outside Top8");
            for (int p : value.at("capture_positions").get<std::vector<int>>())
                if (p < 0 || static_cast<size_t>(p) >= ids.size())
                    throw std::runtime_error("capture position outside prompt");
            const size_t capacity = core.context_capacity();
            if (ids.size() + static_cast<size_t>(max_tokens) > capacity)
                throw std::runtime_error("capture case exceeds context capacity");
            if (value.contains("continuation_ids") && ids.size() + value.at("continuation_ids").size() > capacity)
                throw std::runtime_error("fixed continuation exceeds context capacity");
            const auto eos = c.value("eos_ids", std::vector<int>{});
            capture.begin_case(value);
            auto prefill = [&]() {
                core.reset();
                for (size_t i = 0; i < ids.size(); ++i) {
                    check_cancel();
                    capture.input("prefill", core.position(), ids[i]);
                    core.step(ids[i], i + 1 == ids.size());
                }
            };
            capture.begin_run("baseline_generate");
            prefill();
            std::vector<int> continuation;
            for (int i = 0; i < max_tokens; ++i) {
                check_cancel();
                const auto& v = core.logits();
                const int id = static_cast<int>(std::max_element(v.begin(), v.begin() + core.real_vocab()) - v.begin());
                capture.logits(core, id);
                continuation.push_back(id);
                capture.input("decode", core.position(), id);
                core.step(id, true);
                if (std::find(eos.begin(), eos.end(), id) != eos.end()) break;
            }
            if (value.contains("continuation_ids")) continuation = value.at("continuation_ids").get<std::vector<int>>();
            if (continuation.empty()) throw std::runtime_error("fixed continuation is empty");
            writer.event({{"kind", "continuation"}, {"case", value.at("id")}, {"ids", continuation},
                          {"source", value.contains("continuation_ids") ? "provided" : "baseline_greedy"}});
            if (!c.value("compare", true)) continue;
            json scores = json::object();
            std::vector<std::pair<std::string, int>> runs{{"baseline", -1}};
            for (int rank : value.value("ablation_ranks", std::vector<int>{0, 7})) {
                if (rank < 0 || rank >= 8) throw std::runtime_error("invalid ablation rank");
                runs.push_back({rank == 0 ? "target" : rank == 7 ? "control" : "ablate_rank_" + std::to_string(rank), rank});
            }
            for (const auto& run : runs) {
                capture.begin_run(run.first, run.second);
                prefill();
                double total = 0, first = 0;
                json per_token = json::array();
                for (size_t i = 0; i < continuation.size(); ++i) {
                    check_cancel();
                    const int id = continuation[i];
                    const double lp = log_probability(core, id);
                    if (i == 0) first = std::exp(lp);
                    total += lp;
                    per_token.push_back({{"token_id", id}, {"log_probability", lp}});
                    capture.logits(core, id);
                    capture.input("teacher_force", core.position(), id);
                    core.step(id, i + 1 < continuation.size());
                }
                scores[run.first] = {{"next_token_probability", first}, {"continuation_log_likelihood", total},
                                     {"tokens", per_token}};
            }
            for (const auto& run : runs) {
                if (run.second < 0) continue;
                const auto& name = run.first;
                scores[name]["next_token_probability_delta"] = scores[name]["next_token_probability"].get<double>() - scores["baseline"]["next_token_probability"].get<double>();
                scores[name]["log_likelihood_delta"] = scores[name]["continuation_log_likelihood"].get<double>() - scores["baseline"]["continuation_log_likelihood"].get<double>();
            }
            writer.event({{"kind", "comparison"}, {"case", value.at("id")}, {"source", "derived"}, {"scores", scores}});
        }
        capture.end_run();
        core.set_capture(nullptr);
        writer.finish();
    } catch (...) {
        core.set_capture(nullptr);
        throw;
    }
    return 0;
}
}
