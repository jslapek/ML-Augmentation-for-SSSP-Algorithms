// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
using namespace Rcpp;

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <json.hpp>

#include "common.hpp"
#include "algs/bmssp_stats.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
using distT = long long;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    GraphRow,
    graph_id,
    n,
    m,
    max_dist
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    BMSSPCallRow,
    call_id,
    graph_id,
    parent_call_id,
    depth,
    l,
    B_in,
    B_out,
    S_size,
    U_size,
    P_size,
    W_size,
    status,
    dhat_S_min,
    dhat_S_mean,
    dhat_S_max,
    dhat_S_std,
    edges_relaxed,
    block_pulls,
    block_inserts,
    batch_prepends,
    findpivot_rounds,
    oracle_B_star,
    label_B,
    label_P_eq_S
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    FindPivotsRoundRow,
    call_id,
    round_idx,
    W_i_size,
    W_cumulative,
    relax_attempts,
    relax_successes,
    active_owners,
    top_owner_mass,
    owner_entropy,
    label_P_eq_S
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    PivotSourceRow,
    call_id,
    source_id,
    dhat_s,
    rank_in_S,
    prefix_owner_count,
    final_f_s,
    heavy_label,
    pivot_label
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    BMSSPStats,
    graph_id,
    graphs,
    bmssp_calls,
    findpivots_rounds,
    pivot_sources
)

static std::string read_graph_name_from_cfg(const char* cfg_path) {
    std::ifstream in(cfg_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("Failed to open config file: ") + cfg_path);
    }

    json cfg;
    in >> cfg;

    if (!cfg.contains("graph") || !cfg["graph"].is_string()) {
        throw std::runtime_error("Config must contain a string field 'graph'.");
    }

    return cfg["graph"].get<std::string>();
}

static fs::path make_input_folder_path(const std::string& graph_name) {
    return fs::path("C:/Users/Jakub/Documents/stuff/diss/project/graphs") / (graph_name + "_5k");
}

static fs::path make_graph_path(const fs::path& folder, int graph_idx) {
    return folder / ("graph_" + std::to_string(graph_idx) + ".gr");
}

static fs::path make_output_folder_path(const std::string& graph_name) {
    return fs::path("experiments") / "5k_ml" / graph_name;
}

static fs::path make_output_json_path(const fs::path& out_folder, int graph_idx) {
    return out_folder / ("graph_" + std::to_string(graph_idx) + ".json");
}

// [[Rcpp::export]]
std::string run_5k() {
    const std::string graph_name = read_graph_name_from_cfg("run_class.json");
    const fs::path input_folder = make_input_folder_path(graph_name);
    const fs::path output_folder = make_output_folder_path(graph_name);

    if (!fs::exists(input_folder) || !fs::is_directory(input_folder)) {
        throw std::runtime_error(
            "Input folder does not exist or is not a directory: " + input_folder.string()
        );
    }

    fs::create_directories(output_folder);

    for (int graph_idx = 1; graph_idx <= 5000; ++graph_idx) {
        Rcpp::Rcout << "Processing graph " << graph_idx << " / 5000..." << std::endl;
        const fs::path graph_path = make_graph_path(input_folder, graph_idx);
        if (!fs::exists(graph_path) || !fs::is_regular_file(graph_path)) {
            throw std::runtime_error("Missing graph file: " + graph_path.string());
        }

        auto [adj, m] = readGraph<distT>(graph_path.string());
        (void)m;

        spp_stats::bmssp<distT> solver(adj);
        solver.stats.graph_id = graph_idx;
        solver.prepare_graph(false);
        solver.execute(0);

        json run_json;
        run_json["graph"] = graph_name;
        run_json["graph_index"] = graph_idx;
        run_json["file"] = graph_path.filename().string();
        run_json["stats"] = solver.stats;

        const fs::path output_path = make_output_json_path(output_folder, graph_idx);
        std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("Failed to open output file: " + output_path.string());
        }

        out << run_json.dump(2) << '\n';

        if (!out) {
            throw std::runtime_error("Failed while writing output file: " + output_path.string());
        }
    }

    return "done";
}