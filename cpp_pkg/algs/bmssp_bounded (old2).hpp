#ifndef BMSSP_BOUNDED_HPP
#define BMSSP_BOUNDED_HPP

#include "bmssp.hpp"
#include "utils.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace spp_bounded_opt {

template <typename uniqueDistT>
class batchPQ { // batch priority queue
    template <typename K, typename V>
    using hash_map = ankerl::unordered_dense::map<K, V>;

    using elementT = std::pair<int, uniqueDistT>;

    struct CompareUB {
        template <typename It>
        bool operator()(const std::pair<uniqueDistT, It>& a,
                        const std::pair<uniqueDistT, It>& b) const {
            if (a.first != b.first) return a.first < b.first;
            return std::addressof(*a.second) < std::addressof(*b.second);
        }
    };

    typename std::list<std::list<elementT>>::iterator it_min;

    std::list<std::list<elementT>> D0, D1;
    std::set<std::pair<uniqueDistT, typename std::list<std::list<elementT>>::iterator>, CompareUB> UBs;

    int M = 0;
    int size_ = 0;
    uniqueDistT B;

    hash_map<int, uniqueDistT> actual_value;
    hash_map<int, std::pair<typename std::list<std::list<elementT>>::iterator,
                            typename std::list<elementT>::iterator>>
        where_is0, where_is1;

public:
    batchPQ(int n = 0) : actual_value(n), where_is0(n), where_is1(n) {}

    void initialize(int M_, uniqueDistT B_) {
        M = M_;
        B = B_;
        D0 = {};
        D1 = {std::list<elementT>()};
        it_min = D1.begin();
        UBs = {std::make_pair(B_, D1.begin())};
        size_ = 0;

        actual_value.clear();
        where_is0.clear();
        where_is1.clear();
    }

    int size() const {
        return size_;
    }

    bool empty() const {
        return size_ == 0;
    }

    void insert(uniqueDistT x) {
        uniqueDistT b = x;
        int a = std::get<2>(b);

        auto it_exist = actual_value.find(a);
        bool exist = (it_exist != actual_value.end());

        if (exist && it_exist->second > b) {
            delete_(x);
        } else if (exist) {
            return;
        }

        auto it_UB_block = UBs.lower_bound({b, it_min});
        auto [ub, it_block] = (*it_UB_block);
        (void)ub;

        auto it = it_block->insert(it_block->end(), {a, b});
        where_is1[a] = {it_block, it};
        actual_value[a] = b;

        size_++;

        if ((int)(*it_block).size() > M) {
            split(it_block);
        }
    }

    void batchPrepend(const std::vector<uniqueDistT>& v) {
        std::list<elementT> l;
        for (const auto& x : v) {
            l.push_back({std::get<2>(x), x});
        }
        batchPrepend(l);
    }

    std::pair<uniqueDistT, std::vector<int>> pull() {
        std::vector<elementT> s0, s1;
        s0.reserve(2 * M);
        s1.reserve(M);

        auto it_block = D0.begin();
        while (it_block != D0.end() && (int)s0.size() <= M) {
            for (const auto& x : *it_block) s0.push_back(x);
            ++it_block;
        }

        it_block = D1.begin();
        while (it_block != D1.end() && (int)s1.size() <= M) {
            for (const auto& x : *it_block) s1.push_back(x);
            ++it_block;
        }

        if ((int)(s1.size() + s0.size()) <= M) {
            std::vector<int> ret;
            ret.reserve(s1.size() + s0.size());

            for (auto [a, b] : s0) {
                ret.push_back(a);
                delete_(b);
            }
            for (auto [a, b] : s1) {
                ret.push_back(a);
                delete_(b);
            }

            return {B, ret};
        }

        std::vector<elementT>& l = s0;
        l.insert(l.end(), s1.begin(), s1.end());

        uniqueDistT med = selectKth(l, M);
        std::vector<int> ret;
        ret.reserve(M);

        for (auto [a, b] : l) {
            if (b < med) {
                ret.push_back(a);
                delete_(b);
            }
        }

        return {med, ret};
    }

    inline void erase(int key) {
        if (actual_value.find(key) != actual_value.end()) {
            delete_(uniqueDistT(-1, -1, key, -1));
        }
    }

private:
    void delete_(uniqueDistT x) {
        int a = std::get<2>(x);
        uniqueDistT b = actual_value[a];

        auto it_w = where_is1.find(a);
        if (it_w != where_is1.end()) {
            auto [it_block, it] = it_w->second;
            (*it_block).erase(it);
            where_is1.erase(a);

            if ((*it_block).empty()) {
                auto it_UB_block = UBs.lower_bound({b, it_block});
                if ((*it_UB_block).first != B) {
                    UBs.erase(it_UB_block);
                    D1.erase(it_block);
                }
            }
        } else {
            auto [it_block, it] = where_is0[a];
            (*it_block).erase(it);
            where_is0.erase(a);
            if ((*it_block).empty()) D0.erase(it_block);
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

    void split(std::list<std::list<elementT>>::iterator it_block) {
        int sz = (int)(*it_block).size();

        std::vector<elementT> v((*it_block).begin(), (*it_block).end());
        uniqueDistT med = selectKth(v, sz / 2);

        auto pos = it_block;
        ++pos;

        auto new_block = D1.insert(pos, std::list<elementT>());
        auto it = (*it_block).begin();

        while (it != (*it_block).end()) {
            if ((*it).second >= med) {
                int key = (*it).first;
                (*new_block).push_back(std::move(*it));
                auto it_new = (*new_block).end();
                --it_new;
                where_is1[key] = {new_block, it_new};
                it = (*it_block).erase(it);
            } else {
                ++it;
            }
        }

        uniqueDistT UB1 = {
            std::get<0>(med),
            std::get<1>(med),
            std::get<2>(med),
            std::get<3>(med) - 1
        };

        auto it_lb = UBs.lower_bound({UB1, it_min});
        auto [UB2, aux] = (*it_lb);
        (void)aux;

        UBs.insert({UB1, it_block});
        UBs.insert({UB2, new_block});
        UBs.erase(it_lb);
    }

    void batchPrepend(const std::list<elementT>& l) {
        int sz = (int)l.size();
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

            if (new_block->empty()) D0.erase(new_block);
            return;
        }

        std::vector<elementT> v(l.begin(), l.end());
        uniqueDistT med = selectKth(v, sz / 2);

        std::list<elementT> less, great;
        for (auto [a, b] : l) {
            if (b < med) {
                less.push_back({a, b});
            } else if (b > med) {
                great.push_back({a, b});
            }
        }

        great.push_back({std::get<2>(med), med});
        batchPrepend(great);
        batchPrepend(less);
    }
};

template <typename wT>
class bmssp {
    int n = 0;
    int k = 0;
    int t = 0;
    int l = 0;
    int total_nodes = 0;

    std::vector<std::vector<std::pair<int, wT>>> ori_adj;
    std::vector<std::vector<std::pair<int, wT>>> adj;
    std::vector<wT> d;
    std::vector<int> pred, path_sz;
    std::vector<int> node_map, node_rev_map;
    bool cd_transfomed = false;

public:
    Stats stats;
    const wT oo = std::numeric_limits<wT>::max() / 10;

    bmssp(int n_) : n(n_) {
        ori_adj.assign(n, {});
    }

    bmssp(const auto& adj_) {
        n = (int)adj_.size();
        ori_adj = adj_;
    }

    void addEdge(int a, int b, wT w) {
        ori_adj[a].emplace_back(b, w);
    }

    void set_threshold_schedule(std::vector<wT> thresholds_, bool last_is_final = false) {
        threshold_schedule = std::move(thresholds_);
        last_threshold_is_final = last_is_final;
    }

    void set_final_threshold(wT threshold) {
        threshold_schedule = {threshold};
        last_threshold_is_final = true;
    }

    void clear_threshold_schedule() {
        threshold_schedule.clear();
        last_threshold_is_final = false;
    }

    void prepare_graph(bool exec_constant_degree_trasnformation = false) {
        cd_transfomed = exec_constant_degree_trasnformation;

        std::vector<std::pair<int, int>> tmp_edges(n, {-1, -1});
        for (int i = 0; i < n; i++) {
            std::vector<std::pair<int, wT>> nw_adj;
            nw_adj.reserve(ori_adj[i].size());

            for (auto [j, w] : ori_adj[i]) {
                if (tmp_edges[j].first != i) {
                    nw_adj.emplace_back(j, w);
                    tmp_edges[j] = {i, (int)nw_adj.size() - 1};
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
        } else {
            int cnt = 0;
            std::vector<std::map<int, int>> edge_id(n);

            for (int i = 0; i < n; i++) {
                for (auto [j, w] : ori_adj[i]) {
                    (void)w;
                    if (edge_id[i].find(j) == edge_id[i].end()) {
                        edge_id[i][j] = cnt++;
                        edge_id[j][i] = cnt++;
                    }
                }
            }

            cnt++;
            adj.assign(cnt, {});
            node_map.resize(n);
            node_rev_map.resize(cnt);

            for (int i = 0; i < n; i++) {
                for (auto cur = edge_id[i].begin(); cur != edge_id[i].end(); ++cur) {
                    auto nxt = std::next(cur);
                    if (nxt == edge_id[i].end()) nxt = edge_id[i].begin();
                    adj[cur->second].emplace_back(nxt->second, wT());
                    node_rev_map[cur->second] = i;
                }
            }
            node_rev_map[cnt - 1] = 0;

            for (int i = 0; i < n; i++) {
                for (auto [j, w] : ori_adj[i]) {
                    adj[edge_id[i][j]].emplace_back(edge_id[j][i], w);
                }

                if (!edge_id[i].empty()) {
                    node_map[i] = edge_id[i].begin()->second;
                } else {
                    node_map[i] = cnt - 1;
                }
            }

            ori_adj.clear();
        }

        total_nodes = (int)adj.size();
        d.resize(total_nodes);
        root.resize(total_nodes);
        pred.resize(total_nodes);
        treesz.resize(total_nodes);
        path_sz.resize(total_nodes, 0);
        last_complete_lvl.resize(total_nodes);
        pivot_vis.resize(total_nodes);
        layer_seen.resize(total_nodes);
        u_seen.assign(total_nodes, -1);

        k = std::max(
            1,
            (int)std::floor(std::pow(std::log2((double)std::max(2, total_nodes)), 1.0 / 3.0))
        );
        t = std::max(
            1,
            (int)std::floor(std::pow(std::log2((double)std::max(2, total_nodes)), 2.0 / 3.0))
        );
        l = std::max(
            1,
            (int)std::ceil(std::log2((double)std::max(2, total_nodes)) / t)
        );

        layer_seen_epoch = 0;
        u_seen_epoch = 0;
    }

    std::pair<std::vector<wT>, std::vector<int>> execute(int s) {
        std::fill(d.begin(), d.end(), oo);
        std::fill(last_complete_lvl.begin(), last_complete_lvl.end(), (short int)-1);
        std::fill(pivot_vis.begin(), pivot_vis.end(), -1);
        for (int i = 0; i < (int)pred.size(); i++) pred[i] = i;

        s = toAnyCustomNode(s);
        d[s] = 0;
        path_sz[s] = 0;

        const uniqueDistT inf_dist = {oo, 0, 0, 0};

        std::vector<wT> schedule = normalize_schedule();
        if (schedule.size() == 1) {
            uniqueDistT rootB = inf_dist;
            if (last_threshold_is_final && schedule[0] < oo) {
                rootB = std::min(rootB, thresholdBound(schedule[0]));
            }
            runBMSSPFast((short int)l, rootB, {s});
        } else {
            auto state = makePBMSSPState((short int)l, inf_dist, {s}, (wT)-1, false);

            std::size_t idx = 0;
            while (true) {
                bool is_final_step = last_threshold_is_final && (idx + 1 == schedule.size());
                auto res = resumePBMSSP(std::move(state), schedule[idx], is_final_step);
                if (res.done) break;
                state = std::move(res.state);
                if (idx + 1 == schedule.size()) schedule.push_back(oo);
                ++idx;
            }
        }

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

            int real_path_sz = std::get<1>(getDist(u)) + 1;
            std::vector<int> path(real_path_sz);
            for (int i = real_path_sz - 1; i >= 0; i--) {
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
    template <typename K, typename V>
    using hash_map = ankerl::unordered_dense::map<K, V>;

    inline int toAnyCustomNode(int real_id) const {
        return node_map[real_id];
    }

    inline int customToReal(int id) const {
        return node_rev_map[id];
    }

    int getPred(int u) const {
        int real_u = customToReal(u);
        int dad = u;
        do dad = pred[dad];
        while (customToReal(dad) == real_u && pred[dad] != dad);
        return dad;
    }

    inline int nextUMark() {
        ++u_seen_epoch;
        if (u_seen_epoch == std::numeric_limits<int>::max()) {
            std::fill(u_seen.begin(), u_seen.end(), -1);
            u_seen_epoch = 1;
        }
        return u_seen_epoch;
    }

    inline void appendIfNew(std::vector<int>& U, int mark, int x) {
        if (u_seen[x] != mark) {
            u_seen[x] = mark;
            U.push_back(x);
        }
    }

    struct uniqueDistT : std::tuple<wT, int, int, int> {
        static constexpr wT SCALE = (wT)1e10;
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

    inline uniqueDistT getDist(int u, int v, wT w) const {
        return {d[u] + w, path_sz[u] + 1, v, u};
    }

    inline uniqueDistT getDist(int u) const {
        return {d[u], path_sz[u], u, pred[u]};
    }

    inline uniqueDistT thresholdBound(wT tau) const {
        return {uniqueDistT::sanitize(tau), INT_MAX, INT_MAX, INT_MAX};
    }

    void updateDist(int u, int v, wT w) {
        pred[v] = u;
        d[v] = d[u] + w;
        path_sz[v] = path_sz[u] + 1;
    }

    std::vector<int> root;
    std::vector<short int> treesz;
    int counter_pivot = 0;
    std::vector<int> pivot_vis;
    std::vector<short int> last_complete_lvl;
    std::vector<wT> threshold_schedule;
    bool last_threshold_is_final = false;

    std::vector<int> layer_seen;
    int layer_seen_epoch = 0;
    std::vector<int> u_seen;
    int u_seen_epoch = 0;

    struct SplitHeap {
        hash_map<int, uniqueDistT> value;
        std::priority_queue<uniqueDistT, std::vector<uniqueDistT>, std::greater<uniqueDistT>> active;
        std::priority_queue<uniqueDistT, std::vector<uniqueDistT>, std::greater<uniqueDistT>> dormant;
        wT tau = wT();
        bool active_only = false;

        SplitHeap(int n = 0) : value(n) {}

        void clear(wT new_tau, bool active_only_ = false) {
            value.clear();
            active = {};
            dormant = {};
            tau = new_tau;
            active_only = active_only_;
        }

        void finalize(wT final_tau) {
            if (!active_only) {
                promote(final_tau);
                dormant = {};
                active_only = true;
            }
            tau = final_tau;
        }

        void insert(const uniqueDistT& x) {
            int v = std::get<2>(x);
            auto it = value.find(v);
            if (it != value.end() && !(x < it->second)) return;
            value[v] = x;
            if (active_only || std::get<0>(x) < tau) active.push(x);
            else dormant.push(x);
        }

        void promote(wT new_tau) {
            tau = new_tau;
            if (active_only) {
                cleanActive();
                return;
            }
            cleanDormant();
            while (!dormant.empty()) {
                const auto& x = dormant.top();
                if (std::get<0>(x) >= tau) break;
                active.push(x);
                dormant.pop();
                cleanDormant();
            }
            cleanActive();
        }

        bool activeNonEmpty() {
            cleanActive();
            return !active.empty();
        }

        bool dormantNonEmpty() {
            if (active_only) return false;
            cleanDormant();
            return !dormant.empty();
        }

        uniqueDistT extractMin() {
            cleanActive();
            uniqueDistT x = active.top();
            active.pop();
            auto it = value.find(std::get<2>(x));
            if (it != value.end() && it->second == x) {
                value.erase(it);
            }
            return x;
        }

    private:
        void cleanActive() {
            while (!active.empty()) {
                const auto& x = active.top();
                int v = std::get<2>(x);
                auto it = value.find(v);
                if (it == value.end() || !(it->second == x) || std::get<0>(x) >= tau) {
                    active.pop();
                } else {
                    break;
                }
            }
        }

        void cleanDormant() {
            while (!dormant.empty()) {
                const auto& x = dormant.top();
                int v = std::get<2>(x);
                auto it = value.find(v);
                if (it == value.end() || !(it->second == x)) {
                    dormant.pop();
                } else {
                    break;
                }
            }
        }
    };

    struct SplitBatchPQ {
        batchPQ<uniqueDistT> active;
        hash_map<int, uniqueDistT> dormant_value;
        std::priority_queue<uniqueDistT, std::vector<uniqueDistT>, std::greater<uniqueDistT>> dormant_min;
        int M = 0;
        uniqueDistT B;
        wT tau = wT();
        bool active_only = false;

        SplitBatchPQ(int n = 0) : active(n), dormant_value(n) {}

        void initialize(int M_, uniqueDistT B_, wT tau_, bool active_only_ = false) {
            M = M_;
            B = B_;
            tau = tau_;
            active_only = active_only_;
            active.initialize(M_, B_);
            dormant_value.clear();
            dormant_min = {};
        }

        void finalize(wT final_tau) {
            if (!active_only) {
                promote(final_tau);
                dormant_value.clear();
                dormant_min = {};
                active_only = true;
            }
            tau = final_tau;
        }

        void promote(wT new_tau) {
            if (new_tau <= tau && !active_only) return;
            tau = new_tau;
            if (active_only) return;
            cleanDormant();
            while (!dormant_min.empty()) {
                uniqueDistT x = dormant_min.top();
                if (std::get<0>(x) >= tau) break;
                dormant_min.pop();
                int v = std::get<2>(x);
                auto it = dormant_value.find(v);
                if (it == dormant_value.end() || !(it->second == x)) continue;
                dormant_value.erase(it);
                active.insert(x);
            }
        }

        void insert(const uniqueDistT& x) {
            if (active_only || std::get<0>(x) < tau) {
                active.insert(x);
                return;
            }
            insertDormant(x);
        }

        void batchPrepend(const std::vector<uniqueDistT>& xs) {
            std::vector<uniqueDistT> act;
            act.reserve(xs.size());
            for (const auto& x : xs) {
                if (active_only || std::get<0>(x) < tau) act.push_back(x);
                else insertDormant(x);
            }
            if (!act.empty()) active.batchPrepend(act);
        }

        bool activeNonEmpty() {
            return active.size() > 0;
        }

        bool dormantNonEmpty() {
            if (active_only) return false;
            cleanDormant();
            return !dormant_min.empty();
        }

        bool empty() {
            if (!active_only) cleanDormant();
            return active_only ? active.size() == 0 : (active.size() == 0 && dormant_min.empty());
        }

        std::pair<uniqueDistT, std::vector<int>> pull() {
            return active.pull();
        }

        void erase(int key) {
            active.erase(key);
            if (!active_only) dormant_value.erase(key);
        }

    private:
        void insertDormant(const uniqueDistT& x) {
            int v = std::get<2>(x);
            auto it = dormant_value.find(v);
            if (it != dormant_value.end() && !(x < it->second)) return;
            dormant_value[v] = x;
            dormant_min.push(x);
        }

        void cleanDormant() {
            while (!dormant_min.empty()) {
                const auto& x = dormant_min.top();
                int v = std::get<2>(x);
                auto it = dormant_value.find(v);
                if (it == dormant_value.end() || !(it->second == x)) {
                    dormant_min.pop();
                } else {
                    break;
                }
            }
        }
    };

    struct PBaseCaseState {
        uniqueDistT B;
        int x = -1;
        std::vector<int> U0;
        SplitHeap H;
        bool final_mode = false;

        explicit PBaseCaseState(int n = 0) : H(n) {}
    };

    struct PFindPivotsState {
        uniqueDistT B;
        std::vector<int> S;
        std::vector<int> W;
        std::vector<std::vector<int>> pending;
        std::vector<int> active_buf;
        std::vector<int> remain_buf;
        int next_depth = 0;
        int mark = -1;
    };

    struct PBMSSPState {
        enum class Stage { BASECASE, FIND_PIVOTS, LOOP_READY, LOOP_CHILD };

        Stage stage = Stage::FIND_PIVOTS;
        short int level = 0;
        uniqueDistT B;
        std::vector<int> S;

        std::unique_ptr<PBaseCaseState> base_state;
        std::unique_ptr<PFindPivotsState> find_state;

        SplitBatchPQ D;
        int i = 0;
        uniqueDistT B0;
        uniqueDistT last_complete_B;
        uniqueDistT current_Bi;
        std::vector<int> current_Si;
        std::vector<int> U;
        int U_mark = -1;
        std::vector<int> P;
        std::vector<int> W;
        std::unique_ptr<PBMSSPState> child;
        bool final_mode = false;

        explicit PBMSSPState(int n = 0) : D(n) {}
    };

    struct BaseResult {
        bool done = false;
        uniqueDistT Bprime;
        std::vector<int> U;
        std::unique_ptr<PBaseCaseState> state;
    };

    struct FindResult {
        bool done = false;
        std::vector<int> P;
        std::vector<int> W;
        std::unique_ptr<PFindPivotsState> state;
    };

    struct PBResult {
        bool done = false;
        uniqueDistT Bprime;
        std::vector<int> U;
        std::unique_ptr<PBMSSPState> state;
    };

    std::vector<wT> normalize_schedule() const {
        std::vector<wT> schedule = threshold_schedule;
        if (schedule.empty()) {
            schedule.push_back(oo);
            return schedule;
        }

        for (auto& x : schedule) x = uniqueDistT::sanitize(x);
        std::sort(schedule.begin(), schedule.end());
        schedule.erase(std::unique(schedule.begin(), schedule.end()), schedule.end());
        if (!last_threshold_is_final && schedule.back() < oo) schedule.push_back(oo);
        return schedule;
    }

    std::unique_ptr<PBaseCaseState> makePBaseCaseState(uniqueDistT B, int x, wT tau, bool final_mode) {
        auto st = std::make_unique<PBaseCaseState>(total_nodes);
        st->B = final_mode ? std::min(B, thresholdBound(tau)) : B;
        st->x = x;
        st->U0.clear();
        st->U0.reserve(k + 1);
        st->final_mode = final_mode;
        st->H.clear(tau, final_mode);
        st->H.insert(getDist(x));
        return st;
    }

    BaseResult resumePBaseCase(std::unique_ptr<PBaseCaseState> st, wT tau, bool final_mode) {
        if (final_mode) {
            st->final_mode = true;
            st->B = std::min(st->B, thresholdBound(tau));
            st->H.finalize(tau);
        } else {
            st->H.promote(tau);
        }
        uniqueDistT last_du = getDist(st->x);

        while ((int)st->U0.size() < k + 1) {
            if (!st->H.activeNonEmpty()) {
                if (st->H.dormantNonEmpty()) {
                    return {false, {}, {}, std::move(st)};
                }
                break;
            }

            auto du = st->H.extractMin();
            int u = std::get<2>(du);
            if (du > getDist(u)) continue;

            last_du = du;
            st->U0.push_back(u);

            for (auto [v, w] : adj[u]) {
                auto new_dist = getDist(u, v, w);
                if (new_dist <= getDist(v) && new_dist < st->B) {
                    updateDist(u, v, w);
                    st->H.insert(new_dist);
                }
            }
        }

        if ((int)st->U0.size() <= k) {
            return {true, st->B, std::move(st->U0), nullptr};
        }

        st->U0.pop_back();
        return {true, last_du, std::move(st->U0), nullptr};
    }

    std::unique_ptr<PFindPivotsState> makePFindPivotsState(uniqueDistT B, const std::vector<int>& S) {
        auto st = std::make_unique<PFindPivotsState>();
        st->B = B;
        st->S = S;
        st->W.clear();
        st->W.reserve(std::max<int>((int)S.size(), k * (int)S.size()));
        st->pending.assign(k + 1, {});
        st->pending[0].reserve(S.size());
        st->next_depth = 0;
        st->mark = ++counter_pivot;

        for (int x : S) {
            if (pivot_vis[x] != st->mark) {
                pivot_vis[x] = st->mark;
                st->W.push_back(x);
            }
            root[x] = x;
            treesz[x] = 0;
            st->pending[0].push_back(x);
        }

        return st;
    }

    FindResult resumePFindPivots(std::unique_ptr<PFindPivotsState> st, wT tau, bool final_mode) {
        for (int depth = st->next_depth; depth < k; ++depth) {
            auto& bucket = st->pending[depth];

            if (bucket.empty()) {
                st->next_depth = depth + 1;
                continue;
            }

            auto& active = st->active_buf;
            auto& remain = st->remain_buf;
            active.clear();
            remain.clear();
            active.reserve(bucket.size());
            remain.reserve(bucket.size());

            for (int u : bucket) {
                if (d[u] < tau) active.push_back(u);
                else remain.push_back(u);
            }

            if (active.empty()) {
                if (final_mode) {
                    bucket.clear();
                    st->next_depth = depth + 1;
                    continue;
                }
                bucket.swap(remain);
                st->next_depth = depth;
                return {false, {}, {}, std::move(st)};
            }

            if (final_mode) bucket.clear();
            else bucket.swap(remain);

            ++layer_seen_epoch;
            if (layer_seen_epoch == std::numeric_limits<int>::max()) {
                std::fill(layer_seen.begin(), layer_seen.end(), -1);
                layer_seen_epoch = 0;
                ++layer_seen_epoch;
            }

            auto& next_bucket = st->pending[depth + 1];
            for (int x : next_bucket) layer_seen[x] = layer_seen_epoch;

            for (int u : active) {
                for (auto [v, w] : adj[u]) {
                    auto new_dist = getDist(u, v, w);
                    if (new_dist <= getDist(v)) {
                        updateDist(u, v, w);
                        if (new_dist < st->B) {
                            root[v] = root[u];

                            if (pivot_vis[v] != st->mark) {
                                pivot_vis[v] = st->mark;
                                st->W.push_back(v);
                            }

                            if (layer_seen[v] != layer_seen_epoch) {
                                layer_seen[v] = layer_seen_epoch;
                                next_bucket.push_back(v);
                            }
                        }
                    }
                }
            }

            if ((int)st->W.size() > k * (int)st->S.size()) {
                return {true, st->S, st->W, nullptr};
            }

            if (!bucket.empty()) {
                if (final_mode) {
                    bucket.clear();
                } else {
                    st->next_depth = depth;
                    return {false, {}, {}, std::move(st)};
                }
            }

            st->next_depth = depth + 1;
        }

        for (int u : st->S) treesz[u] = 0;
        for (int u : st->W) treesz[root[u]]++;

        std::vector<int> P;
        P.reserve(st->W.size() / std::max(1, k));
        for (int u : st->S) {
            if (treesz[u] >= k) P.push_back(u);
        }

        return {true, std::move(P), std::move(st->W), nullptr};
    }

    std::pair<uniqueDistT, std::vector<int>> runBaseCaseFast(uniqueDistT B, int x) {
        std::priority_queue<uniqueDistT, std::vector<uniqueDistT>, std::greater<uniqueDistT>> H;
        std::vector<int> U0;
        U0.reserve(k + 1);
        uniqueDistT last_du = getDist(x);

        H.push(last_du);
        while (!H.empty() && (int)U0.size() < k + 1) {
            auto du = H.top();
            H.pop();
            int u = std::get<2>(du);
            if (du > getDist(u)) continue;

            last_du = du;
            U0.push_back(u);

            for (auto [v, w] : adj[u]) {
                auto new_dist = getDist(u, v, w);
                if (new_dist <= getDist(v) && new_dist < B) {
                    updateDist(u, v, w);
                    H.push(new_dist);
                }
            }
        }

        if ((int)U0.size() <= k) return {B, std::move(U0)};
        U0.pop_back();
        return {last_du, std::move(U0)};
    }

    std::pair<std::vector<int>, std::vector<int>> runFindPivotsFast(uniqueDistT B, const std::vector<int>& S) {
        std::vector<int> W;
        W.reserve(std::max<int>((int)S.size(), k * (int)S.size()));
        std::vector<std::vector<int>> pending(k + 1);
        pending[0].reserve(S.size());
        int mark = ++counter_pivot;

        for (int x : S) {
            if (pivot_vis[x] != mark) {
                pivot_vis[x] = mark;
                W.push_back(x);
            }
            root[x] = x;
            treesz[x] = 0;
            pending[0].push_back(x);
        }

        for (int depth = 0; depth < k; ++depth) {
            auto& bucket = pending[depth];
            if (bucket.empty()) continue;

            ++layer_seen_epoch;
            if (layer_seen_epoch == std::numeric_limits<int>::max()) {
                std::fill(layer_seen.begin(), layer_seen.end(), -1);
                layer_seen_epoch = 1;
            }

            auto& next_bucket = pending[depth + 1];
            for (int x : next_bucket) layer_seen[x] = layer_seen_epoch;

            for (int u : bucket) {
                for (auto [v, w] : adj[u]) {
                    auto new_dist = getDist(u, v, w);
                    if (new_dist <= getDist(v)) {
                        updateDist(u, v, w);
                        if (new_dist < B) {
                            root[v] = root[u];

                            if (pivot_vis[v] != mark) {
                                pivot_vis[v] = mark;
                                W.push_back(v);
                            }

                            if (layer_seen[v] != layer_seen_epoch) {
                                layer_seen[v] = layer_seen_epoch;
                                next_bucket.push_back(v);
                            }
                        }
                    }
                }
            }

            if ((int)W.size() > k * (int)S.size()) {
                return {S, W};
            }
        }

        for (int u : S) treesz[u] = 0;
        for (int u : W) treesz[root[u]]++;

        std::vector<int> P;
        P.reserve(W.size() / std::max(1, k));
        for (int u : S) {
            if (treesz[u] >= k) P.push_back(u);
        }

        return {std::move(P), std::move(W)};
    }

    std::pair<uniqueDistT, std::vector<int>> runBMSSPFast(short int level,
                                                          uniqueDistT B,
                                                          const std::vector<int>& S) {
        if (level == 0) return runBaseCaseFast(B, S[0]);

        auto [P, W] = runFindPivotsFast(B, S);

        const long long batch_size = (1ll << ((level - 1) * t));
        SplitBatchPQ D(total_nodes);
        D.initialize((int)batch_size, B, oo);
        for (int p : P) D.insert(getDist(p));

        uniqueDistT B0 = B;
        for (int p : P) B0 = std::min(B0, getDist(p));
        uniqueDistT last_complete_B = B0;

        const long long quota = 1ll * k * (1ll << (level * t));
        std::vector<int> U;
        U.reserve((size_t)std::min<long long>(quota, total_nodes));
        int U_mark = nextUMark();

        while ((long long)U.size() < quota && !D.empty()) {
            auto pulled = D.pull();
            uniqueDistT current_Bi = pulled.first;
            std::vector<int> current_Si = std::move(pulled.second);

            auto [complete_B, nw_complete] = runBMSSPFast((short int)(level - 1), current_Bi, current_Si);
            for (int x : nw_complete) appendIfNew(U, U_mark, x);

            std::vector<uniqueDistT> can_prepend;
            can_prepend.reserve(nw_complete.size() * 5 + current_Si.size());

            for (int u : nw_complete) {
                D.erase(u);
                last_complete_lvl[u] = level;

                for (auto [v, w] : adj[u]) {
                    auto new_dist = getDist(u, v, w);
                    if (new_dist <= getDist(v)) {
                        updateDist(u, v, w);
                        if (current_Bi <= new_dist && new_dist < B) {
                            D.insert(new_dist);
                        } else if (complete_B <= new_dist && new_dist < current_Bi) {
                            can_prepend.emplace_back(new_dist);
                        }
                    }
                }
            }

            for (int x : current_Si) {
                if (complete_B <= getDist(x)) can_prepend.emplace_back(getDist(x));
            }

            D.batchPrepend(can_prepend);
            last_complete_B = complete_B;
        }

        uniqueDistT retB = D.empty() ? B : last_complete_B;
        for (int x : W) {
            if (last_complete_lvl[x] != level && getDist(x) < retB) {
                appendIfNew(U, U_mark, x);
            }
        }

        return {retB, std::move(U)};
    }

    std::unique_ptr<PBMSSPState> makePBMSSPState(short int level,
                                                 uniqueDistT B,
                                                 const std::vector<int>& S,
                                                 wT tau,
                                                 bool final_mode) {
        auto st = std::make_unique<PBMSSPState>(total_nodes);
        st->level = level;
        st->B = final_mode ? std::min(B, thresholdBound(tau)) : B;
        st->S = S;
        st->final_mode = final_mode;

        if (level == 0) {
            st->stage = PBMSSPState::Stage::BASECASE;
            st->base_state = makePBaseCaseState(st->B, S[0], tau, final_mode);
        } else {
            st->stage = PBMSSPState::Stage::FIND_PIVOTS;
            st->find_state = makePFindPivotsState(st->B, S);
        }

        return st;
    }

    PBResult resumePBMSSP(std::unique_ptr<PBMSSPState> st, wT tau, bool final_mode) {
        st->final_mode = st->final_mode || final_mode;
        if (final_mode) st->B = std::min(st->B, thresholdBound(tau));

        if (st->stage == PBMSSPState::Stage::BASECASE) {
            auto base_res = resumePBaseCase(std::move(st->base_state), tau, st->final_mode);
            if (base_res.done) {
                return {true, base_res.Bprime, std::move(base_res.U), nullptr};
            }
            st->base_state = std::move(base_res.state);
            return {false, {}, {}, std::move(st)};
        }

        if (st->stage == PBMSSPState::Stage::FIND_PIVOTS) {
            auto fp_res = resumePFindPivots(std::move(st->find_state), tau, st->final_mode);
            if (!fp_res.done) {
                st->find_state = std::move(fp_res.state);
                return {false, {}, {}, std::move(st)};
            }

            st->P = std::move(fp_res.P);
            st->W = std::move(fp_res.W);
            st->find_state.reset();

            const long long batch_size = (1ll << ((st->level - 1) * t));
            st->D.initialize((int)batch_size, st->B, tau, st->final_mode);
            for (int p : st->P) st->D.insert(getDist(p));

            st->i = 0;
            st->B0 = st->B;
            for (int p : st->P) st->B0 = std::min(st->B0, getDist(p));
            st->last_complete_B = st->B0;
            st->U.clear();
            st->U_mark = nextUMark();
            st->stage = PBMSSPState::Stage::LOOP_READY;
        }

        const long long quota = 1ll * k * (1ll << (st->level * t));
        if (st->U.empty()) st->U.reserve((size_t)std::min<long long>(quota, total_nodes));

        while ((long long)st->U.size() < quota) {
            if (st->stage == PBMSSPState::Stage::LOOP_CHILD) {
                auto child_res = resumePBMSSP(std::move(st->child), tau, st->final_mode);
                if (!child_res.done) {
                    st->child = std::move(child_res.state);
                    return {false, {}, {}, std::move(st)};
                }

                uniqueDistT complete_B = child_res.Bprime;
                std::vector<int> nw_complete = std::move(child_res.U);

                for (int x : nw_complete) appendIfNew(st->U, st->U_mark, x);

                std::vector<uniqueDistT> can_prepend;
                can_prepend.reserve(nw_complete.size() * 5 + st->current_Si.size());

                for (int u : nw_complete) {
                    st->D.erase(u);
                    last_complete_lvl[u] = st->level;

                    for (auto [v, w] : adj[u]) {
                        auto new_dist = getDist(u, v, w);
                        if (new_dist <= getDist(v)) {
                            updateDist(u, v, w);
                            if (st->current_Bi <= new_dist && new_dist < st->B) {
                                st->D.insert(new_dist);
                            } else if (complete_B <= new_dist && new_dist < st->current_Bi) {
                                can_prepend.emplace_back(new_dist);
                            }
                        }
                    }
                }

                for (int x : st->current_Si) {
                    if (complete_B <= getDist(x)) can_prepend.emplace_back(getDist(x));
                }

                st->D.batchPrepend(can_prepend);
                st->last_complete_B = complete_B;
                st->child.reset();
                st->stage = PBMSSPState::Stage::LOOP_READY;
                continue;
            }

            if (st->final_mode) st->D.finalize(tau);
            else st->D.promote(tau);

            if (!st->D.activeNonEmpty()) {
                if (st->D.dormantNonEmpty()) {
                    return {false, {}, {}, std::move(st)};
                }
                break;
            }

            st->i++;
            auto pulled = st->D.pull();
            st->current_Bi = pulled.first;
            st->current_Si = std::move(pulled.second);
            uniqueDistT childB = st->current_Bi;
            if (st->final_mode) childB = std::min(childB, thresholdBound(tau));
            st->child = makePBMSSPState((short int)(st->level - 1), childB, st->current_Si, tau, st->final_mode);
            st->stage = PBMSSPState::Stage::LOOP_CHILD;
        }

        uniqueDistT retB = st->D.empty() ? st->B : st->last_complete_B;
        for (int x : st->W) {
            if (last_complete_lvl[x] != st->level && getDist(x) < retB) {
                appendIfNew(st->U, st->U_mark, x);
            }
        }

        return {true, retB, std::move(st->U), nullptr};
    }
};

} // namespace spp_bounded

#endif