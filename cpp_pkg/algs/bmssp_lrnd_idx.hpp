#pragma once

// Learned-index variant of bmssp_timed.hpp
// - All timing instrumentation removed.
// - Keeps the .stats attribute on bmssp.
// - batchPQ frontier insertion uses a selectable learned index for UB lower_bound:
//   method = "alex" | "pgm" | "radix" (fallbacks to std::set if unknown)

#include "bmssp.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ALEX (header-only; expects numeric keys)
#include "../structures/alex_map.h"          // microsoft/ALEX/src/core/alex_map.h

// PGM-index (header-only)
#include "../structures/pgm_index.hpp"   // gvinciguerra/PGM-index

// RadixSpline (header-only)
#include "../structures/builder.h"        // learnedsystems/RadixSpline/include/rs/builder.h
#include "../structures/radix_spline.h"   // learnedsystems/RadixSpline/include/rs/radix_spline.h

namespace spp_lrnd_idx {

namespace detail {
inline std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
}

template<typename uniqueDistT>
class batchPQ { // batch priority queue
    template<typename K, typename V>
    using hash_map = ankerl::unordered_dense::map<K, V>;
    using elementT = std::pair<int, uniqueDistT>;

    using BlockIt = typename std::list<std::list<elementT>>::iterator;

    // Comparator used for std::set of UBs.
    struct CompareUB {
        template <typename It>
        bool operator()(const std::pair<uniqueDistT, It>& a,
                        const std::pair<uniqueDistT, It>& b) const {
            if (a.first != b.first) return a.first < b.first;
            return std::addressof(*a.second) < std::addressof(*b.second);
        }
    };

    // ------------------------------------------------------------------
    // Core PQ state
    BlockIt it_min;
    std::list<std::list<elementT>> D0, D1;
    std::set<std::pair<uniqueDistT, BlockIt>, CompareUB> UBs;

    int M = 0;
    int size_ = 0;
    uniqueDistT B;

    hash_map<int, uniqueDistT> actual_value;
    hash_map<int, std::pair<BlockIt, typename std::list<elementT>::iterator>> where_is0, where_is1;

    // ------------------------------------------------------------------
    // Learned-index selection and auxiliary snapshots
    enum class UBMethod : uint8_t { StdSet, Alex, Pgm, Radix };
    UBMethod ub_method_ = UBMethod::StdSet;

    struct UBEntry {
        uniqueDistT ub;
        BlockIt     block;
        std::uintptr_t addr; // tie-break key for equal ub
        long long    d;       // projection: get<0>(ub)
    };

    std::vector<UBEntry> ubv_;          // snapshot of UBs in sorted (CompareUB) order
    std::vector<long long> d_unique_;   // strictly increasing unique distances (get<0>(ub))
    std::vector<uint32_t> run_start_;   // start indices into ubv_ for each d_unique_
    std::vector<uint64_t> du_unique_;   // ordered-encoded distances (for RadixSpline)

    // Method-specific indexes over d_unique_
    alex::AlexMap<long long, uint32_t> alex_d2run_;                 // d -> index into d_unique_ (NOT run_start)
    static constexpr int PGM_EPS = 64;                               // tune if desired
    pgm::PGMIndex<long long, PGM_EPS> pgm_;                           // over d_unique_
    rs::RadixSpline<uint64_t> rs_;                                    // over du_unique_

public:
    // Keep these (bmssp_timed.hpp aggregates them into stats), but we no longer time.
    double snip_split = 0.0;
    double snip_lower_bound = 0.0;
    double snip_block_insertion = 0.0;
    double snip_membership_check = 0.0;
    double snip_deletion = 0.0;

    explicit batchPQ(int n)
        : actual_value(n), where_is0(n), where_is1(n) {}

    // Initialize with the UB indexing method: "alex" | "pgm" | "radix".
    void initialize(int M_, uniqueDistT B_, const std::string& method = "pgm") {
        M = M_;
        B = B_;
        D0.clear();
        D1 = { std::list<elementT>() };
        size_ = 0;

        // Reset snips (kept for compatibility)
        snip_split = snip_lower_bound = snip_block_insertion = snip_membership_check = snip_deletion = 0.0;

        actual_value.clear();
        where_is0.clear();
        where_is1.clear();

        it_min = D1.begin();
        UBs.clear();
        UBs.insert({B, D1.begin()});

        set_ub_method_(method);
        rebuild_ub_snapshot_();
    }

    int size() const { return size_; }

    void insert(uniqueDistT x) {
        uniqueDistT b = x;
        int a = std::get<2>(b);

        // membership / decrease-key
        auto it_exist = actual_value.find(a);
        bool exist = (it_exist != actual_value.end());
        if (exist && it_exist->second > b) {
            delete_(x);
        } else if (exist) {
            return;
        }

        // Route to the right UB block using the selected method.
        BlockIt it_block = ub_lower_bound_block_(b);

        // Insert into that block.
        auto it = it_block->insert(it_block->end(), {a, b});
        where_is1[a] = {it_block, it};
        actual_value[a] = b;
        size_++;

        if (static_cast<int>((*it_block).size()) > M) {
            split(it_block);
        }
    }

    void batchPrepend(const std::vector<uniqueDistT>& v) {
        std::list<elementT> l;
        for (auto x : v) {
            l.push_back({std::get<2>(x), x});
        }
        batchPrepend(l);
    }

    std::pair<uniqueDistT, std::vector<int>> pull() {
        std::vector<elementT> s0, s1;
        s0.reserve(2 * M);
        s1.reserve(M);

        auto it_block = D0.begin();
        while (it_block != D0.end() && static_cast<int>(s0.size()) <= M) {
            for (const auto& x : *it_block) s0.push_back(x);
            ++it_block;
        }

        it_block = D1.begin();
        while (it_block != D1.end() && static_cast<int>(s1.size()) <= M) {
            for (const auto& x : *it_block) s1.push_back(x);
            ++it_block;
        }

        if (static_cast<int>(s1.size() + s0.size()) <= M) {
            std::vector<int> ret;
            ret.reserve(s1.size() + s0.size());
            for (auto [aa, bb] : s0) {
                ret.push_back(aa);
                delete_({bb});
            }
            for (auto [aa, bb] : s1) {
                ret.push_back(aa);
                delete_({bb});
            }
            return {B, ret};
        }

        std::vector<elementT>& l = s0;
        l.insert(l.end(), s1.begin(), s1.end());

        uniqueDistT med = selectKth(l, M);
        std::vector<int> ret;
        ret.reserve(M);
        for (auto [aa, bb] : l) {
            if (bb < med) {
                ret.push_back(aa);
                delete_({bb});
            }
        }
        return {med, ret};
    }

    inline void erase(int key) {
        if (actual_value.find(key) != actual_value.end())
            delete_({-1, -1, key, -1});
    }

private:
    // ----------------------- Learned UB routing helpers -----------------------

    static inline uint64_t i64_to_u64_ordered(long long x) {
        return uint64_t(x) ^ 0x8000'0000'0000'0000ULL;
    }

    static inline std::uintptr_t block_addr_(const BlockIt& it) {
        return reinterpret_cast<std::uintptr_t>(std::addressof(*it));
    }

    void set_ub_method_(const std::string& method) {
        std::string m = detail::to_lower(method);
        if (m == "alex") ub_method_ = UBMethod::Alex;
        else if (m == "pgm") ub_method_ = UBMethod::Pgm;
        else if (m == "radix" || m == "radixspline") ub_method_ = UBMethod::Radix;
        else ub_method_ = UBMethod::StdSet;
    }

    // Build ubv_ (sorted snapshot of UBs) and method-specific indexes.
    // Called only when UBs changes (split / empty block removal) and at initialize.
    void rebuild_ub_snapshot_() {
        ubv_.clear();
        ubv_.reserve(UBs.size());

        d_unique_.clear();
        run_start_.clear();
        du_unique_.clear();

        uint32_t idx = 0;
        long long last_d = 0;
        bool have_last_d = false;

        for (const auto& [ub, blk] : UBs) {
            // NOTE: this assumes get<0>(uniqueDistT) is integral-like and fits in long long.
            long long d = static_cast<long long>(std::get<0>(ub));
            ubv_.push_back(UBEntry{ub, blk, block_addr_(blk), d});

            if (!have_last_d || d != last_d) {
                d_unique_.push_back(d);
                run_start_.push_back(idx);
                du_unique_.push_back(i64_to_u64_ordered(d));
                last_d = d;
                have_last_d = true;
            }
            ++idx;
        }

        // Build method-specific index over d_unique_.
        if (ub_method_ == UBMethod::Alex) {
            alex_d2run_.clear();
            for (uint32_t i = 0; i < d_unique_.size(); ++i) {
                alex_d2run_.insert(d_unique_[i], i);
            }
        } else if (ub_method_ == UBMethod::Pgm) {
            // PGMIndex supports strictly increasing keys; d_unique_ is strictly increasing.
            pgm_ = pgm::PGMIndex<long long, PGM_EPS>(d_unique_.begin(), d_unique_.end());
        } else if (ub_method_ == UBMethod::Radix) {
            // RadixSpline requires uint64_t keys (we feed ordered-encoded signed distances).
            rs::Builder<uint64_t> b(du_unique_.front(), du_unique_.back());
            for (auto k : du_unique_) b.AddKey(k);
            rs_ = b.Finalize();
        }
    }

    // Compare UBEntry vs search key (ub, addr_key).
    static inline bool ubentry_less_key_(const UBEntry& e, const std::pair<uniqueDistT, std::uintptr_t>& k) {
        if (e.ub != k.first) return e.ub < k.first;
        return e.addr < k.second;
    }

    // Exact lower_bound on ubv_ within [lo,hi)
    BlockIt exact_lb_ubv_window_(const uniqueDistT& b, size_t lo, size_t hi) const {
        // Match the std::set query key {b, it_min} tie-break behavior.
        const std::uintptr_t addr_key = (it_min == D1.end()) ? 0u : reinterpret_cast<std::uintptr_t>(std::addressof(*it_min));
        const std::pair<uniqueDistT, std::uintptr_t> key{b, addr_key};
        auto it = std::lower_bound(ubv_.begin() + lo, ubv_.begin() + hi, key, ubentry_less_key_);
        if (it == ubv_.begin() + hi) {
            // Safety fallback: full search
            it = std::lower_bound(ubv_.begin(), ubv_.end(), key, ubentry_less_key_);
        }
        return it->block;
    }

    // Return the block iterator corresponding to UBs.lower_bound({b, it_min}).
    BlockIt ub_lower_bound_block_(const uniqueDistT& b) {
        if (ub_method_ == UBMethod::StdSet) {
            auto it = UBs.lower_bound({b, it_min});
            return it->second;
        }

        // Snapshot-based learned routing.
        if (ubv_.empty() || d_unique_.empty()) {
            auto it = UBs.lower_bound({b, it_min});
            return it->second;
        }

        long long d = static_cast<long long>(std::get<0>(b));

        // Step 1: find the first distance group d' >= d (index in d_unique_)
        size_t di = 0;

        if (ub_method_ == UBMethod::Alex) {
            auto it = alex_d2run_.lower_bound(d);
            if (it == alex_d2run_.end()) {
                // Should not happen if sentinel exists; fallback to last.
                return ubv_.back().block;
            }
            di = static_cast<size_t>(it.payload());
        } else if (ub_method_ == UBMethod::Pgm) {
            auto ap = pgm_.search(d);
            size_t lo = static_cast<size_t>(ap.lo);
            size_t hi = static_cast<size_t>(ap.hi);
            lo = std::min(lo, d_unique_.size());
            hi = std::min(hi, d_unique_.size());
            if (lo > hi) std::swap(lo, hi);
            auto it = std::lower_bound(d_unique_.begin() + lo, d_unique_.begin() + hi, d);
            if (it == d_unique_.begin() + hi) {
                it = std::lower_bound(d_unique_.begin(), d_unique_.end(), d);
            }
            if (it == d_unique_.end()) return ubv_.back().block;
            di = static_cast<size_t>(it - d_unique_.begin());
        } else { // RadixSpline
            uint64_t du = i64_to_u64_ordered(d);
            rs::SearchBound bound = rs_.GetSearchBound(du);
            size_t lo = std::min<size_t>(bound.begin, du_unique_.size());
            size_t hi = std::min<size_t>(bound.end, du_unique_.size());
            if (lo > hi) std::swap(lo, hi);
            auto it = std::lower_bound(du_unique_.begin() + lo, du_unique_.begin() + hi, du);
            if (it == du_unique_.begin() + hi) {
                it = std::lower_bound(du_unique_.begin(), du_unique_.end(), du);
            }
            if (it == du_unique_.end()) return ubv_.back().block;
            di = static_cast<size_t>(it - du_unique_.begin());
        }

        // Step 2: translate distance-group index -> [ub_lo, ub_hi) in ubv_
        size_t ub_lo = static_cast<size_t>(run_start_[di]);
        size_t ub_hi = (di + 1 < run_start_.size()) ? static_cast<size_t>(run_start_[di + 1])
                                                    : ubv_.size();
        long long dprime = d_unique_[di];

        // If dprime > d, then any UB in this group is >= b (because first tuple component dominates).
        if (dprime > d) {
            return ubv_[ub_lo].block;
        }

        // dprime == d: must refine using full tuple order within that run.
        return exact_lb_ubv_window_(b, ub_lo, ub_hi);
    }

    // ----------------------------- Core helpers ------------------------------

    void delete_(uniqueDistT x) {
        int a = std::get<2>(x);
        uniqueDistT b = actual_value[a];

        auto it_w = where_is1.find(a);
        if (it_w != where_is1.end()) {
            auto [it_block, it] = it_w->second;

            (*it_block).erase(it);
            where_is1.erase(a);

            if ((*it_block).size() == 0) {
                auto it_UB_block = UBs.lower_bound({b, it_block});
                if ((*it_UB_block).first != B) {
                    UBs.erase(it_UB_block);
                    D1.erase(it_block);
                    it_min = D1.begin();
                    rebuild_ub_snapshot_();
                }
            }
        } else {
            auto [it_block, it] = where_is0[a];
            (*it_block).erase(it);
            where_is0.erase(a);
            if ((*it_block).size() == 0) D0.erase(it_block);
        }

        actual_value.erase(a);
        size_--;
    }

    uniqueDistT selectKth(std::vector<elementT>& v, int k) {
        const auto comparator = [](const auto& a, const auto& b) {
            return a.second < b.second;
        };
        miniselect::floyd_rivest_select(v.begin(), v.begin() + k, v.end(), comparator);
        return v[k].second;
    }

    void split(BlockIt it_block) {
        int sz = static_cast<int>((*it_block).size());
        std::vector<elementT> v((*it_block).begin(), (*it_block).end());
        uniqueDistT med = selectKth(v, (sz / 2));

        auto pos = it_block;
        ++pos;

        auto new_block = D1.insert(pos, std::list<elementT>());
        auto it = (*it_block).begin();

        while (it != (*it_block).end()) {
            if ((*it).second >= med) {
                (*new_block).push_back(std::move(*it));
                auto it_new = (*new_block).end();
                --it_new;
                where_is1[(*it).first] = {new_block, it_new};

                it = (*it_block).erase(it);
            } else {
                ++it;
            }
        }

        // Updating UBs
        uniqueDistT UB1 = {std::get<0>(med), std::get<1>(med), std::get<2>(med), std::get<3>(med) - 1};
        auto it_lb = UBs.lower_bound({UB1, it_min});
        auto UB2 = (*it_lb).first;

        UBs.insert({UB1, it_block});
        UBs.insert({UB2, new_block});
        UBs.erase(it_lb);

        it_min = D1.begin();
        rebuild_ub_snapshot_();
    }

    void batchPrepend(const std::list<elementT>& l) {
        int sz = static_cast<int>(l.size());
        if (sz == 0) return;

        if (sz <= M) {
            D0.push_front(std::list<elementT>());
            auto new_block = D0.begin();

            for (const auto& x : l) {
                auto it = actual_value.find(x.first);
                bool exist = (it != actual_value.end());

                if (exist && it->second > x.second) {
                    delete_(x.second);
                } else if (exist) {
                    continue;
                }

                (*new_block).push_back(x);
                auto it_new = (*new_block).end();
                --it_new;
                where_is0[x.first] = {new_block, it_new};
                actual_value[x.first] = x.second;
                size_++;
            }

            if (new_block->size() == 0) D0.erase(new_block);
            return;
        }

        std::vector<elementT> v(l.begin(), l.end());
        uniqueDistT med = selectKth(v, sz / 2);

        std::list<elementT> less, great;
        for (auto [a, b] : l) {
            if (b < med) less.push_back({a, b});
            else if (b > med) great.push_back({a, b});
        }
        great.push_back({std::get<2>(med), med});

        batchPrepend(great);
        batchPrepend(less);
    }
};

//////////////////////////////////////////////////////

template<typename wT>
class bmssp {
    // Base Attributes
    int n, k, t, l;

    std::vector<std::vector<std::pair<int, wT>>> ori_adj;
    std::vector<std::vector<std::pair<int, wT>>> adj;
    std::vector<wT> d;
    std::vector<int> pred, path_sz;

    std::vector<int> node_map, node_rev_map;

    bool cd_transfomed;

    // Frontier learned-index method (propagated to batchPQ::initialize)
    std::string frontier_index_method_ = "pgm";

public:
    Stats stats;
    const wT oo = std::numeric_limits<wT>::max() / 10;

    // Choose the frontier UB lookup backend at construction time.
    // method: "alex" | "pgm" | "radix" (unknown -> std::set fallback)
    explicit bmssp(int n_, std::string frontier_index_method = "pgm")
        : n(n_), frontier_index_method_(std::move(frontier_index_method)) {
        ori_adj.assign(n, {});
    }

    // C++20 abbreviated template constructor (matches original style).
    explicit bmssp(const auto& adj_, std::string frontier_index_method = "pgm")
        : frontier_index_method_(std::move(frontier_index_method)) {
        n = static_cast<int>(adj_.size());
        ori_adj = adj_;
    }

    void addEdge(int a, int b, wT w) {
        ori_adj[a].emplace_back(b, w);
    }

    void set_frontier_index_method(const std::string& method) {
        frontier_index_method_ = method;
    }

    // if the graph already has constant degree, prepage_graph(false)
    // else, prepage_graph(true)
    void prepare_graph(bool exec_constant_degree_trasnformation = false) {
        cd_transfomed = exec_constant_degree_trasnformation;

        // erase duplicated edges
        std::vector<std::pair<int, int>> tmp_edges(n, {-1, -1});
        for (int i = 0; i < n; i++) {
            std::vector<std::pair<int, wT>> nw_adj;
            nw_adj.reserve(ori_adj[i].size());
            for (auto [j, w] : ori_adj[i]) {
                if (tmp_edges[j].first != i) {
                    nw_adj.emplace_back(j, w);
                    tmp_edges[j] = {i, static_cast<int>(nw_adj.size()) - 1};
                } else {
                    int id = tmp_edges[j].second;
                    nw_adj[id].second = std::min(nw_adj[id].second, w);
                }
            }
            ori_adj[i] = std::move(nw_adj);
        }
        tmp_edges.clear();

        if (!exec_constant_degree_trasnformation) {
            adj = std::move(ori_adj);
            node_map.resize(n);
            node_rev_map.resize(n);

            for (int i = 0; i < n; i++) {
                node_map[i] = i;
                node_rev_map[i] = i;
            }

            k = static_cast<int>(std::floor(std::pow(std::log2(n), 1.0 / 3.0)));
            t = static_cast<int>(std::floor(std::pow(std::log2(n), 2.0 / 3.0)));
        } else {
            int cnt = 0;
            std::vector<std::map<int, int>> edge_id(n);
            for (int i = 0; i < n; i++) {
                for (auto [j, w] : ori_adj[i]) {
                    if (edge_id[i].find(j) == edge_id[i].end()) {
                        edge_id[i][j] = cnt++;
                        edge_id[j][i] = cnt++;
                    }
                }
            }

            cnt++;
            adj.assign(cnt, {});
            node_map.resize(cnt);
            node_rev_map.resize(cnt);

            for (int i = 0; i < n; i++) { // create 0-weight cycles
                for (auto cur = edge_id[i].begin(); cur != edge_id[i].end(); cur++) {
                    auto nxt = std::next(cur);
                    if (nxt == edge_id[i].end()) nxt = edge_id[i].begin();
                    adj[cur->second].emplace_back(nxt->second, wT());
                    node_rev_map[cur->second] = i;
                }
            }
            for (int i = 0; i < n; i++) { // add edges
                for (auto [j, w] : ori_adj[i]) {
                    adj[edge_id[i][j]].emplace_back(edge_id[j][i], w);
                }
                if (edge_id[i].size()) {
                    node_map[i] = edge_id[i].begin()->second;
                } else {
                    node_map[i] = cnt - 1;
                }
            }

            ori_adj.clear();
        }

        d.resize(adj.size());
        root.resize(adj.size());
        pred.resize(adj.size());
        treesz.resize(adj.size());
        path_sz.resize(adj.size(), 0);
        last_complete_lvl.resize(adj.size());
        pivot_vis.resize(adj.size());

        k = static_cast<int>(std::floor(std::pow(std::log2(adj.size()), 1.0 / 3.0)));
        t = static_cast<int>(std::floor(std::pow(std::log2(adj.size()), 2.0 / 3.0)));
        l = static_cast<int>(std::ceil(std::log2(adj.size()) / t));

        // Allocate one batchPQ per recursion level; each needs hash tables sized to |V|.
        Ds.clear();
        Ds.reserve(l);
        for (int i = 0; i < l; ++i) {
            Ds.emplace_back(static_cast<int>(adj.size()));
        }
    }

    std::pair<std::vector<wT>, std::vector<int>> execute(int s) {

        std::fill(d.begin(), d.end(), oo);
        std::fill(last_complete_lvl.begin(), last_complete_lvl.end(), -1);
        std::fill(pivot_vis.begin(), pivot_vis.end(), -1);
        for (int i = 0; i < static_cast<int>(pred.size()); i++) pred[i] = i;

        s = toAnyCustomNode(s);
        d[s] = 0;
        path_sz[s] = 0;

        const int level = static_cast<int>(std::ceil(std::log2(adj.size()) / t));
        const uniqueDistT inf_dist = {oo, 0, 0, 0};

        bmsspRec(level, inf_dist, {s});

        if (!cd_transfomed) {
            return {d, pred};
        }

        std::vector<wT> ret_distance(n);
        std::vector<int> ret_pred(n);
        for (int i = 0; i < n; i++) {
            ret_distance[i] = d[toAnyCustomNode(i)];
            ret_pred[i] = customToReal(getPred(toAnyCustomNode(i)));
        }
        return {ret_distance, ret_pred};
    }

    std::vector<int> get_shortest_path(int real_u, const std::vector<int>& real_pred) {
        if (!cd_transfomed) {
            int u = real_u;
            if (d[u] == oo) return {};

            int psz = std::get<1>(getDist(u)) + 1;
            std::vector<int> path(psz);
            for (int i = psz - 1; i >= 0; i--) {
                path[i] = u;
                u = pred[u];
            }
            return path;
        }

        int u = real_u;
        if (d[toAnyCustomNode(u)] == oo) return {};

        int max_path_sz = std::get<1>(getDist(toAnyCustomNode(u))) + 1;
        std::vector<int> path;
        path.reserve(max_path_sz);

        int oldu;
        do {
            path.push_back(u);
            oldu = u;
            u = real_pred[u];
        } while (u != oldu);

        std::reverse(path.begin(), path.end());
        return path;
    }

private:
    inline int toAnyCustomNode(int real_id) { return node_map[real_id]; }
    inline int customToReal(int id) { return node_rev_map[id]; }

    int getPred(int u) {
        int real_u = customToReal(u);

        int dad = u;
        do dad = pred[dad];
        while (customToReal(dad) == real_u && pred[dad] != dad);

        return dad;
    }

    // Unique distances helpers: Assumption 2.1
    struct uniqueDistT : std::tuple<wT, int, int, int> {
        static constexpr wT SCALE = 1e10;
        static constexpr wT SCALE_INV = ((wT)1.0) / SCALE;

        uniqueDistT() = default;
        static inline wT sanitize(wT w) {
            if constexpr (std::is_floating_point_v<wT>) {
                return std::round(w * SCALE) * SCALE_INV;
            }
            return w;
        }
        uniqueDistT(wT w, int i1, int i2, int i3)
            : std::tuple<wT, int, int, int>(sanitize(w), i1, i2, i3) {}
    };

    inline uniqueDistT getDist(int u, int v, wT w) { return {d[u] + w, path_sz[u] + 1, v, u}; }
    inline uniqueDistT getDist(int u) { return {d[u], path_sz[u], u, pred[u]}; }

    void updateDist(int u, int v, wT w) {
        pred[v] = u;
        d[v] = d[u] + w;
        path_sz[v] = path_sz[u] + 1;
    }

    // ===================================================================
    std::vector<int> root;
    std::vector<short int> treesz;

    int counter_pivot = 0;
    std::vector<int> pivot_vis;

    std::pair<std::vector<int>, std::vector<int>> findPivots(uniqueDistT B, const std::vector<int>& S) {
        counter_pivot++;

        std::vector<int> vis;
        vis.reserve(2 * k * S.size());

        for (int x : S) {
            vis.push_back(x);
            pivot_vis[x] = counter_pivot;
        }

        std::vector<int> active = S;
        for (int x : S) root[x] = x, treesz[x] = 0;

        for (int i = 1; i <= k; i++) {
            std::vector<int> nw_active;
            nw_active.reserve(active.size() * 4);
            for (int u : active) {
                for (auto [v, w] : adj[u]) {
                    if (getDist(u, v, w) <= getDist(v)) {
                        updateDist(u, v, w);
                        if (getDist(v) < B) {
                            root[v] = root[u];
                            nw_active.push_back(v);
                        }
                    }
                }
            }
            for (const auto& x : nw_active) {
                if (pivot_vis[x] != counter_pivot) {
                    pivot_vis[x] = counter_pivot;
                    vis.push_back(x);
                }
            }
            if (vis.size() > static_cast<size_t>(k) * S.size()) {
                return {S, vis};
            }
            active = std::move(nw_active);
        }

        std::vector<int> P;
        P.reserve(vis.size() / k);
        for (int u : vis) treesz[root[u]]++;
        for (int u : S) if (treesz[u] >= k) P.push_back(u);

        return {P, vis};
    }

    std::pair<uniqueDistT, std::vector<int>> baseCase(uniqueDistT B, int x) {
        std::vector<int> complete;
        complete.reserve(k + 1);

        std::priority_queue<uniqueDistT, std::vector<uniqueDistT>, std::greater<uniqueDistT>> heap;
        heap.push(getDist(x));
        while (!heap.empty() && complete.size() < static_cast<size_t>(k + 1)) {
            auto du = heap.top();
            int u = std::get<2>(du);
            heap.pop();

            if (du > getDist(u)) continue;

            complete.push_back(u);
            for (auto [v, w] : adj[u]) {
                auto new_dist = getDist(u, v, w);
                auto old_dist = getDist(v);
                if (new_dist <= old_dist && new_dist < B) {
                    updateDist(u, v, w);
                    heap.push(new_dist);
                }
            }
        }
        if (complete.size() <= static_cast<size_t>(k)) return {B, complete};

        uniqueDistT nB = getDist(complete.back());
        complete.pop_back();
        return {nB, complete};
    }

    std::vector<batchPQ<uniqueDistT>> Ds;
    std::vector<short int> last_complete_lvl;

    std::pair<uniqueDistT, std::vector<int>> bmsspRec(short int l, uniqueDistT B, const std::vector<int>& S) {
        if (l == 0) {
            return baseCase(B, S[0]);
        }

        auto [P, bellman_vis] = findPivots(B, S);

        const long long batch_size = (1ll << ((l - 1) * t));
        auto& D = Ds[l - 1];
        D.initialize(static_cast<int>(batch_size), B, frontier_index_method_);

        for (int p : P) D.insert(getDist(p));

        uniqueDistT last_complete_B = B;
        for (int p : P) last_complete_B = std::min(last_complete_B, getDist(p));

        std::vector<int> complete;
        const long long quota = k * (1ll << (l * t));
        complete.reserve(static_cast<size_t>(quota) + bellman_vis.size());

        while (static_cast<long long>(complete.size()) < quota && D.size()) {
            auto [trying_B, miniS] = D.pull();
            auto [complete_B, nw_complete] = bmsspRec(l - 1, trying_B, miniS);

            complete.insert(complete.end(), nw_complete.begin(), nw_complete.end());

            std::vector<uniqueDistT> can_prepend;
            can_prepend.reserve(nw_complete.size() * 5 + miniS.size());

            for (int u : nw_complete) {
                D.erase(u);

                last_complete_lvl[u] = l;
                for (auto [v, w] : adj[u]) {
                    auto new_dist = getDist(u, v, w);
                    if (new_dist <= getDist(v)) {
                        updateDist(u, v, w);
                        if (trying_B <= new_dist && new_dist < B) {
                            D.insert(new_dist);
                        } else if (complete_B <= new_dist && new_dist < trying_B) {
                            can_prepend.emplace_back(new_dist);
                        }
                    }
                }
            }

            for (int x : miniS) {
                if (complete_B <= getDist(x)) can_prepend.emplace_back(getDist(x));
            }

            D.batchPrepend(can_prepend);
            last_complete_B = complete_B;
        }

        uniqueDistT retB;
        if (D.size() == 0) retB = B;
        else retB = last_complete_B;

        for (int x : bellman_vis) {
            if (last_complete_lvl[x] != l && getDist(x) < retB) {
                complete.push_back(x);
            }
        }

        // Keep these aggregations (snips remain 0 without timers)
        stats.snip_lower_bound += D.snip_lower_bound;
        stats.snip_split += D.snip_split;
        stats.snip_block_insertion += D.snip_block_insertion;
        stats.snip_membership_check += D.snip_membership_check;
        stats.snip_deletion += D.snip_deletion;

        return {retB, complete};
    }
};

} // namespace spp_lrnd_idx
