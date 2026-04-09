#include "bmssp.hpp"
#include "utils_bmssp_stats.hpp"

namespace spp_stats {

template<typename uniqueDistT>
class batchPQ {
    template<typename K, typename V>
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

    int M = 0, size_ = 0;
    uniqueDistT B;

    hash_map<int, uniqueDistT> actual_value;
    hash_map<int, std::pair<typename std::list<std::list<elementT>>::iterator,
                            typename std::list<elementT>::iterator>> where_is0, where_is1;

public:
    batchPQ(int n) : actual_value(n), where_is0(n), where_is1(n) {}

    void initialize(int M_, uniqueDistT B_) {
        M = M_;
        B = B_;
        D0 = {};
        D1 = {std::list<elementT>()};
        UBs = {std::make_pair(B_, D1.begin())};
        size_ = 0;

        actual_value.clear();
        where_is0.clear();
        where_is1.clear();
    }

    int size() const {
        return size_;
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

        if ((*it_block).size() > static_cast<size_t>(M)) {
            split(it_block);
        }
    }

    void batchPrepend(const std::vector<uniqueDistT>& v) {
        std::list<elementT> l;
        for (auto x : v) l.push_back({std::get<2>(x), x});
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

        if (static_cast<int>(s0.size() + s1.size()) <= M) {
            std::vector<int> ret;
            ret.reserve(s0.size() + s1.size());
            for (auto [a, b] : s0) {
                ret.push_back(a);
                delete_({b});
            }
            for (auto [a, b] : s1) {
                ret.push_back(a);
                delete_({b});
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
                delete_({b});
            }
        }
        return {med, ret};
    }

    inline void erase(int key) {
        if (actual_value.find(key) != actual_value.end()) {
            delete_({-1, -1, key, -1});
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
        int sz = static_cast<int>((*it_block).size());

        std::vector<elementT> v((*it_block).begin(), (*it_block).end());
        uniqueDistT med = selectKth(v, sz / 2);

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

        uniqueDistT UB1 = {std::get<0>(med), std::get<1>(med), std::get<2>(med), std::get<3>(med) - 1};
        auto it_lb = UBs.lower_bound({UB1, it_min});
        auto [UB2, aux] = (*it_lb);
        (void)aux;

        UBs.insert({UB1, it_block});
        UBs.insert({UB2, new_block});
        UBs.erase(it_lb);
    }

    void batchPrepend(const std::list<elementT>& l) {
        int sz = static_cast<int>(l.size());
        if (sz == 0) return;

        if (sz <= M) {
            D0.push_front(std::list<elementT>());
            auto new_block = D0.begin();

            for (auto& x : l) {
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

//////////////////////////////////////////////////////

template<typename wT>
class bmssp {
    int n = 0, k = 0, t = 0, l = 0;
    int original_n = 0;
    long long original_m = 0;

    std::vector<std::vector<std::pair<int, wT>>> ori_adj;
    std::vector<std::vector<std::pair<int, wT>>> adj;
    std::vector<wT> d;
    std::vector<int> pred, path_sz;

    std::vector<int> node_map, node_rev_map;
    bool cd_transfomed = false;

    int next_call_id = 0;
    std::vector<int> call_stack;

public:
    BMSSPStats stats;
    const wT oo = std::numeric_limits<wT>::max() / 10;

    bmssp(int n_) : n(n_), original_n(n_) {
        ori_adj.assign(n, {});
    }

    bmssp(const auto& adj_) {
        n = static_cast<int>(adj_.size());
        original_n = n;
        ori_adj = adj_;
    }

    void addEdge(int a, int b, wT w) {
        ori_adj[a].emplace_back(b, w);
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
                    tmp_edges[j] = {i, static_cast<int>(nw_adj.size()) - 1};
                } else {
                    int id = tmp_edges[j].second;
                    nw_adj[id].second = std::min(nw_adj[id].second, w);
                }
            }
            ori_adj[i] = std::move(nw_adj);
        }
        tmp_edges.clear();

        original_m = 0;
        for (const auto& out : ori_adj) original_m += static_cast<long long>(out.size());

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
            node_map.resize(cnt);
            node_rev_map.resize(cnt);

            for (int i = 0; i < n; i++) {
                for (auto cur = edge_id[i].begin(); cur != edge_id[i].end(); ++cur) {
                    auto nxt = std::next(cur);
                    if (nxt == edge_id[i].end()) nxt = edge_id[i].begin();
                    adj[cur->second].emplace_back(nxt->second, wT());
                    node_rev_map[cur->second] = i;
                }
            }
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

        d.resize(adj.size());
        root.resize(adj.size());
        pred.resize(adj.size());
        treesz.resize(adj.size());
        path_sz.resize(adj.size(), 0);
        last_complete_lvl.resize(adj.size());
        pivot_vis.resize(adj.size());
        k = std::floor(std::pow(std::log2(static_cast<double>(adj.size())), 1.0 / 3.0));
        t = std::floor(std::pow(std::log2(static_cast<double>(adj.size())), 2.0 / 3.0));
        if (t <= 0) t = 1;
        l = std::ceil(std::log2(static_cast<double>(adj.size())) / t);
        Ds.assign(l, adj.size());
    }

    std::pair<std::vector<wT>, std::vector<int>> execute(int s) {
        stats.clear_tables();
        next_call_id = 0;
        call_stack.clear();

        std::fill(d.begin(), d.end(), oo);
        std::fill(last_complete_lvl.begin(), last_complete_lvl.end(), -1);
        std::fill(pivot_vis.begin(), pivot_vis.end(), -1);
        for (int i = 0; i < static_cast<int>(pred.size()); i++) pred[i] = i;

        s = toAnyCustomNode(s);
        d[s] = 0;
        path_sz[s] = 0;

        const int top_l = std::ceil(std::log2(static_cast<double>(adj.size())) / t);
        const uniqueDistT inf_dist = {oo, 0, 0, 0};
        bmsspRec(top_l, inf_dist, {s}, 0);

        std::vector<wT> ret_distance;
        std::vector<int> ret_pred;

        if (!cd_transfomed) {
            ret_distance = d;
            ret_pred = pred;
        } else {
            ret_distance.resize(original_n);
            ret_pred.resize(original_n);
            for (int i = 0; i < original_n; i++) {
                ret_distance[i] = d[toAnyCustomNode(i)];
                ret_pred[i] = customToReal(getPred(toAnyCustomNode(i)));
            }
        }

        double max_dist = 0.0;
        for (wT x : ret_distance) {
            if (x < oo / 2) max_dist = std::max(max_dist, static_cast<double>(x));
        }
        stats.graphs.push_back(GraphRow{stats.graph_id, original_n, original_m, max_dist});

        return {ret_distance, ret_pred};
    }

    std::vector<int> get_shortest_path(int real_u, const std::vector<int>& real_pred) {
        if (!cd_transfomed) {
            int u = real_u;
            if (d[u] == oo) return {};

            int path_sz_local = std::get<1>(getDist(u)) + 1;
            std::vector<int> path(path_sz_local);
            for (int i = path_sz_local - 1; i >= 0; i--) {
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
    inline int toAnyCustomNode(int real_id) {
        return node_map[real_id];
    }

    inline int customToReal(int id) {
        return node_rev_map[id];
    }

    int getPred(int u) {
        int real_u = customToReal(u);

        int dad = u;
        do dad = pred[dad];
        while (customToReal(dad) == real_u && pred[dad] != dad);

        return dad;
    }

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

    inline uniqueDistT getDist(int u, int v, wT w) {
        return {d[u] + w, path_sz[u] + 1, v, u};
    }

    inline uniqueDistT getDist(int u) {
        return {d[u], path_sz[u], u, pred[u]};
    }

    void updateDist(int u, int v, wT w) {
        pred[v] = u;
        d[v] = d[u] + w;
        path_sz[v] = path_sz[u] + 1;
    }

    static double dist_to_scalar(const uniqueDistT& x) {
        return static_cast<double>(std::get<0>(x));
    }

    std::tuple<double, double, double, double> summarize_S(const std::vector<int>& S) {
        if (S.empty()) return {0.0, 0.0, 0.0, 0.0};

        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        double sum = 0.0;
        for (int x : S) {
            double val = dist_to_scalar(getDist(x));
            mn = std::min(mn, val);
            mx = std::max(mx, val);
            sum += val;
        }
        double mean = sum / static_cast<double>(S.size());
        double var = 0.0;
        for (int x : S) {
            double val = dist_to_scalar(getDist(x));
            double diff = val - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(S.size());
        return {mn, mean, mx, std::sqrt(var)};
    }

    int begin_call(short int level, const uniqueDistT& B, const std::vector<int>& S, int depth) {
        auto [mn, mean, mx, sd] = summarize_S(S);
        int parent_call_id = call_stack.empty() ? -1 : call_stack.back();
        int call_id = next_call_id++;

        BMSSPCallRow row;
        row.call_id = call_id;
        row.graph_id = stats.graph_id;
        row.parent_call_id = parent_call_id;
        row.depth = depth;
        row.l = static_cast<int>(level);
        row.B_in = dist_to_scalar(B);
        row.S_size = static_cast<int>(S.size());
        row.dhat_S_min = mn;
        row.dhat_S_mean = mean;
        row.dhat_S_max = mx;
        row.dhat_S_std = sd;
        stats.bmssp_calls.push_back(row);
        call_stack.push_back(call_id);
        return call_id;
    }

    BMSSPCallRow& current_call_row() {
        return stats.bmssp_calls.back();
    }

    void end_call(int call_id, const uniqueDistT& B_out, int U_size, int P_size, int W_size, bool label_P_eq_S) {
        BMSSPCallRow& row = stats.bmssp_calls.back();
        (void)call_id;
        row.B_out = dist_to_scalar(B_out);
        row.U_size = U_size;
        row.P_size = P_size;
        row.W_size = W_size;
        row.status = (row.B_out >= row.B_in);
        row.label_B = row.B_out;
        row.label_P_eq_S = label_P_eq_S;
        call_stack.pop_back();
    }

    void count_successful_relaxation() {
        current_call_row().edges_relaxed++;
    }

    // ===================================================================
    std::vector<int> root;
    std::vector<short int> treesz;

    int counter_pivot = 0;
    std::vector<int> pivot_vis;

    struct FindPivotsResult {
        std::vector<int> P;
        std::vector<int> vis;
        bool returned_P_eq_S = false;
    };

    FindPivotsResult findPivots(uniqueDistT B, const std::vector<int>& S, int call_id) {
        static constexpr int PREFIX_ROUND = 1;
        counter_pivot++;

        std::vector<int> vis;
        vis.reserve(2 * std::max(1, k) * std::max(1, static_cast<int>(S.size())));

        ankerl::unordered_dense::map<int, int> source_index(static_cast<size_t>(S.size() * 2 + 1));
        for (int i = 0; i < static_cast<int>(S.size()); ++i) source_index[S[i]] = i;

        std::vector<int> owner_counts(S.size(), 0);
        std::vector<int> prefix_owner_counts(S.size(), 0);

        for (int x : S) {
            vis.push_back(x);
            pivot_vis[x] = counter_pivot;
            root[x] = x;
            treesz[x] = 0;
            owner_counts[source_index[x]] = 1;
        }

        std::vector<int> active = S;
        std::vector<size_t> round_row_ids;
        bool returned_P_eq_S = false;

        for (int i = 1; i <= k; i++) {
            std::vector<int> nw_active;
            nw_active.reserve(active.size() * 4);
            long long relax_attempts = 0;
            long long relax_successes = 0;

            for (int u : active) {
                for (auto [v, w] : adj[u]) {
                    relax_attempts++;
                    if (getDist(u, v, w) <= getDist(v)) {
                        updateDist(u, v, w);
                        count_successful_relaxation();
                        relax_successes++;
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
                    auto it = source_index.find(root[x]);
                    if (it != source_index.end()) owner_counts[it->second]++;
                }
            }

            if (i == PREFIX_ROUND) prefix_owner_counts = owner_counts;
            if (k < PREFIX_ROUND) prefix_owner_counts = owner_counts;

            int active_owners = 0;
            int top_owner = 0;
            for (int c : owner_counts) {
                if (c > 0) active_owners++;
                top_owner = std::max(top_owner, c);
            }

            FindPivotsRoundRow rr;
            rr.call_id = call_id;
            rr.round_idx = i;
            rr.W_i_size = static_cast<int>(nw_active.size());
            rr.W_cumulative = static_cast<int>(vis.size());
            rr.relax_attempts = relax_attempts;
            rr.relax_successes = relax_successes;
            rr.active_owners = active_owners;
            rr.top_owner_mass = vis.empty() ? 0.0 : static_cast<double>(top_owner) / static_cast<double>(vis.size());
            rr.owner_entropy = BMSSPStats::entropy_from_counts(owner_counts);
            stats.findpivots_rounds.push_back(rr);
            round_row_ids.push_back(stats.findpivots_rounds.size() - 1);
            current_call_row().findpivot_rounds++;

            if (static_cast<int>(vis.size()) > k * static_cast<int>(S.size())) {
                returned_P_eq_S = true;
                for (size_t idx : round_row_ids) stats.findpivots_rounds[idx].label_P_eq_S = true;

                std::vector<std::pair<double, int>> ranked;
                ranked.reserve(S.size());
                for (int s : S) ranked.push_back({dist_to_scalar(getDist(s)), s});
                std::sort(ranked.begin(), ranked.end());
                ankerl::unordered_dense::map<int, int> rank_of_s(static_cast<size_t>(S.size() * 2 + 1));
                for (int pos = 0; pos < static_cast<int>(ranked.size()); ++pos) rank_of_s[ranked[pos].second] = pos + 1;

                for (int s : S) {
                    int idx = source_index[s];
                    PivotSourceRow ps;
                    ps.call_id = call_id;
                    ps.source_id = s;
                    ps.dhat_s = dist_to_scalar(getDist(s));
                    ps.rank_in_S = rank_of_s[s];
                    ps.prefix_owner_count = prefix_owner_counts[idx] == 0 ? owner_counts[idx] : prefix_owner_counts[idx];
                    ps.final_f_s = owner_counts[idx];
                    ps.heavy_label = owner_counts[idx] >= k;
                    ps.pivot_label = true;
                    stats.pivot_sources.push_back(ps);
                }

                current_call_row().P_size = static_cast<int>(S.size());
                current_call_row().W_size = static_cast<int>(vis.size());
                return {S, vis, true};
            }

            active = std::move(nw_active);
        }

        if (prefix_owner_counts.empty()) prefix_owner_counts = owner_counts;
        if (k > 0 && std::all_of(prefix_owner_counts.begin(), prefix_owner_counts.end(), [](int x) { return x == 0; })) {
            prefix_owner_counts = owner_counts;
        }

        std::vector<int> P;
        P.reserve(vis.size() / std::max(1, k));
        for (int u : vis) treesz[root[u]]++;
        for (int u : S) if (treesz[u] >= k) P.push_back(u);

        std::vector<std::pair<double, int>> ranked;
        ranked.reserve(S.size());
        for (int s : S) ranked.push_back({dist_to_scalar(getDist(s)), s});
        std::sort(ranked.begin(), ranked.end());
        ankerl::unordered_dense::map<int, int> rank_of_s(static_cast<size_t>(S.size() * 2 + 1));
        for (int pos = 0; pos < static_cast<int>(ranked.size()); ++pos) rank_of_s[ranked[pos].second] = pos + 1;

        ankerl::unordered_dense::set<int> inP(static_cast<size_t>(P.size() * 2 + 1));
        for (int x : P) inP.insert(x);

        for (int s : S) {
            int idx = source_index[s];
            PivotSourceRow ps;
            ps.call_id = call_id;
            ps.source_id = s;
            ps.dhat_s = dist_to_scalar(getDist(s));
            ps.rank_in_S = rank_of_s[s];
            ps.prefix_owner_count = prefix_owner_counts[idx] == 0 ? owner_counts[idx] : prefix_owner_counts[idx];
            ps.final_f_s = owner_counts[idx];
            ps.heavy_label = owner_counts[idx] >= k;
            ps.pivot_label = inP.find(s) != inP.end();
            stats.pivot_sources.push_back(ps);
        }

        current_call_row().P_size = static_cast<int>(P.size());
        current_call_row().W_size = static_cast<int>(vis.size());
        return {P, vis, returned_P_eq_S};
    }

    std::pair<uniqueDistT, std::vector<int>> baseCase(uniqueDistT B, int x) {
        std::vector<int> complete;
        complete.reserve(k + 1);

        std::priority_queue<uniqueDistT, std::vector<uniqueDistT>, std::greater<uniqueDistT>> heap;
        heap.push(getDist(x));
        while (!heap.empty() && static_cast<int>(complete.size()) < k + 1) {
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
                    count_successful_relaxation();
                    heap.push(new_dist);
                }
            }
        }
        if (static_cast<int>(complete.size()) <= k) return {B, complete};

        uniqueDistT nB = getDist(complete.back());
        complete.pop_back();
        return {nB, complete};
    }

    std::vector<batchPQ<uniqueDistT>> Ds;
    std::vector<short int> last_complete_lvl;

    std::pair<uniqueDistT, std::vector<int>> bmsspRec(short int l, uniqueDistT B, const std::vector<int>& S, int depth) {
        int call_id = begin_call(l, B, S, depth);

        if (l == 0) {
            auto x = baseCase(B, S[0]);
            end_call(call_id, x.first, static_cast<int>(x.second.size()), 0, 0, false);
            return x;
        }

        auto fp = findPivots(B, S, call_id);
        auto& row = current_call_row();

        const long long batch_size = (1ll << ((l - 1) * t));
        auto& D = Ds[l - 1];
        D.initialize(static_cast<int>(batch_size), B);

        for (int p : fp.P) {
            D.insert(getDist(p));
            row.block_inserts++;
        }

        uniqueDistT last_complete_B = B;
        for (int p : fp.P) last_complete_B = std::min(last_complete_B, getDist(p));

        std::vector<int> complete;
        const long long quota = k * (1ll << (l * t));
        complete.reserve(static_cast<size_t>(quota) + fp.vis.size());
        while (static_cast<long long>(complete.size()) < quota && D.size()) {
            auto [trying_B, miniS] = D.pull();
            row.block_pulls++;

            auto [complete_B, nw_complete] = bmsspRec(l - 1, trying_B, miniS, depth + 1);
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
                        count_successful_relaxation();
                        if (trying_B <= new_dist && new_dist < B) {
                            D.insert(new_dist);
                            row.block_inserts++;
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
            row.batch_prepends++;

            last_complete_B = complete_B;
        }

        uniqueDistT retB = D.size() == 0 ? B : last_complete_B;
        for (int x : fp.vis) {
            if (last_complete_lvl[x] != l && getDist(x) < retB) {
                complete.push_back(x);
            }
        }

        end_call(call_id, retB, static_cast<int>(complete.size()), static_cast<int>(fp.P.size()), static_cast<int>(fp.vis.size()), fp.returned_P_eq_S);
        return {retB, complete};
    }
};

} // namespace spp_stats
