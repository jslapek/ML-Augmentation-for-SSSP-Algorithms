#include bmssp.hpp

namespace spp_timed {

template<typename uniqueDistT>
using batchPQ = spp_expected::batchPQ<uniqueDistT>;

template<typename wT>
class bmssp { // bmssp class
    int n, k, t, l;

    std::vector<std::vector<std::pair<int, wT>>> ori_adj;
    std::vector<std::vector<std::pair<int, wT>>> adj;
    std::vector<wT> d;
    std::vector<int> pred, path_sz;

    std::vector<int> node_map, node_rev_map;
    
    bool cd_transfomed;

public:
    const wT oo = std::numeric_limits<wT>::max() / 10;
    bmssp(int n_): n(n_) {
        ori_adj.assign(n, {});
    }
    bmssp(const auto &adj) {
        n = adj.size();
        ori_adj = adj;
    }
    
    void addEdge(int a, int b, wT w) {
        ori_adj[a].emplace_back(b, w);
    }

    // if the graph already has constant degree, prepage_graph(false)
    // else, prepage_graph(true)
    void prepare_graph(bool exec_constant_degree_trasnformation = false) {
        cd_transfomed = exec_constant_degree_trasnformation;
        // erase duplicated edges
        std::vector<std::pair<int, int>> tmp_edges(n, {-1, -1});
        for(int i = 0; i < n; i++) {
            std::vector<std::pair<int, wT>> nw_adj;
            nw_adj.reserve(ori_adj[i].size());
            for(auto [j, w]: ori_adj[i]) {
                if(tmp_edges[j].first != i) {
                    nw_adj.emplace_back(j, w);
                    tmp_edges[j] = {i, nw_adj.size() - 1};
                } else {
                    int id = tmp_edges[j].second;
                    nw_adj[id].second = std::min(nw_adj[id].second, w);
                }
            }
            ori_adj[i] = move(nw_adj);
        }
        tmp_edges.clear();

        if(exec_constant_degree_trasnformation == false) {
            adj = move(ori_adj);
            node_map.resize(n);
            node_rev_map.resize(n);
            
            for(int i = 0; i < n; i++) {
                node_map[i] = i;
                node_rev_map[i] = i;
            }

            k = floor(pow(log2(n), 1.0 / 3.0));
            t = floor(pow(log2(n), 2.0 / 3.0));
        } else { // Make the graph become constant degree
            int cnt = 0;
            std::vector<std::map<int, int>> edge_id(n);
            for(int i = 0; i < n; i++) {
                for(auto [j, w]: ori_adj[i]) {
                    if(edge_id[i].find(j) == edge_id[i].end()) {
                        edge_id[i][j] = cnt++;
                        edge_id[j][i] = cnt++;
                    }
                }
            }

            cnt++;
            adj.assign(cnt, {});
            node_map.resize(cnt);
            node_rev_map.resize(cnt);
    
            for(int i = 0; i < n; i++) { // create 0-weight cycles
                for(auto cur = edge_id[i].begin(); cur != edge_id[i].end(); cur++) {
                    auto nxt = next(cur);
                    if(nxt == edge_id[i].end()) nxt = edge_id[i].begin();
                    adj[cur->second].emplace_back(nxt->second, wT());
                    node_rev_map[cur->second] = i;
                }
            }
            for(int i = 0; i < n; i++) { // add edges
                for(auto [j, w]: ori_adj[i]) {
                    adj[edge_id[i][j]].emplace_back(edge_id[j][i], w);
                }
                if(edge_id[i].size()) {
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
        k = floor(pow(log2(adj.size()), 1.0 / 3.0));
        t = floor(pow(log2(adj.size()), 2.0 / 3.0));
        l = ceil(log2(adj.size()) / t);
        Ds.assign(l, adj.size());
    }

    std::pair<std::vector<wT>, std::vector<int>> execute(int s) {
        fill(d.begin(), d.end(), oo);
        fill(last_complete_lvl.begin(), last_complete_lvl.end(), -1);
        fill(pivot_vis.begin(), pivot_vis.end(), -1);
        for(int i = 0; i < pred.size(); i++) pred[i] = i;
        
        s = toAnyCustomNode(s);
        d[s] = 0;
        path_sz[s] = 0;
        
        const int l = ceil(log2(adj.size()) / t);
        const uniqueDistT inf_dist = {oo, 0, 0, 0};
        bmsspRec(l, inf_dist, {s});
        
        if(!cd_transfomed) {
            return {d, pred};
        } else {
            std::vector<wT> ret_distance(n);
            std::vector<int> ret_pred(n);
            for(int i = 0; i < n; i++) {
                ret_distance[i] = d[toAnyCustomNode(i)];
                ret_pred[i] = customToReal(getPred(toAnyCustomNode(i)));
            }
            return {ret_distance, ret_pred};
        }
    }

    std::vector<int> get_shortest_path(int real_u, const std::vector<int> &real_pred) {
        if(!cd_transfomed) {
            int u = real_u;
            if(d[u] == oo) return {};

            int path_sz = get<1>(getDist(u)) + 1;
            std::vector<int> path(path_sz);
            for(int i = path_sz - 1; i >= 0; i--) {
                path[i] = u;
                u = pred[u];
            }
            return path; // {source, ..., real_u}
        } else {
            int u = real_u;
            if(d[toAnyCustomNode(u)] == oo) return {};

            int max_path_sz = get<1>(getDist(toAnyCustomNode(u))) + 1;
            std::vector<int> path;
            path.reserve(max_path_sz);

            int oldu;
            do {
                path.push_back(u);
                oldu = u;
                u = real_pred[u];
            } while(u != oldu);

            reverse(path.begin(), path.end());
            return path; // {source, ..., real_u}
        }
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
        while(customToReal(dad) == real_u && pred[dad] != dad);

        return dad;
    }

    template<typename T>
    bool isUnique(const std::vector<T> &v) {
        auto v2 = v;
        sort(v2.begin(), v2.end());
        v2.erase(unique(v2.begin(), v2.end()), v2.end());
        return v2.size() == v.size();
    }

    // Unique distances helpers: Assumption 2.1
    struct uniqueDistT : std::tuple<wT, int, int, int> {
        static constexpr wT SCALE = 1e10;
        static constexpr wT SCALE_INV = ((wT) 1.0) / SCALE; 

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

    // ===================================================================
    std::vector<int> root;
    std::vector<short int> treesz;

    int counter_pivot = 0;
    std::vector<int> pivot_vis;
    std::pair<std::vector<int>, std::vector<int>> findPivots(uniqueDistT B, const std::vector<int> &S) { // Algorithm 1
        counter_pivot++;

        std::vector<int> vis;
        vis.reserve(2 * k * S.size());

        for(int x: S) {
            vis.push_back(x);
            pivot_vis[x] = counter_pivot;
        }

        std::vector<int> active = S;
        for(int x: S) root[x] = x, treesz[x] = 0;
        for(int i = 1; i <= k; i++) {
            std::vector<int> nw_active;
            nw_active.reserve(active.size() * 4);
            for(int u: active) {
                for(auto [v, w]: adj[u]) {
                    if(getDist(u, v, w) <= getDist(v)) {
                        updateDist(u, v, w);
                        if(getDist(v) < B) {
                            root[v] = root[u];
                            nw_active.push_back(v);
                        }
                    }
                }
            }
            for(const auto &x: nw_active) {
                if(pivot_vis[x] != counter_pivot) {
                    pivot_vis[x] = counter_pivot;
                    vis.push_back(x);
                }
            }
            if(vis.size() > k * S.size()) {
                return {S, vis};
            }
            active = move(nw_active);
        }

        std::vector<int> P;
        P.reserve(vis.size() / k);
        for(int u: vis) treesz[root[u]]++;
        for(int u: S) if(treesz[u] >= k) P.push_back(u);
        
        // assert(P.size() <= vis.size() / k);
        return {P, vis};
    }
 
    std::pair<uniqueDistT, std::vector<int>> baseCase(uniqueDistT B, int x) { // Algorithm 2
        std::vector<int> complete;
        complete.reserve(k + 1);
 
        std::priority_queue<uniqueDistT, std::vector<uniqueDistT>, std::greater<uniqueDistT>> heap;
        heap.push(getDist(x));
        while(heap.empty() == false && complete.size() < k + 1) {
            auto du = heap.top();
            int u = get<2>(du);
            heap.pop();

            if(du > getDist(u)) continue;

            complete.push_back(u);
            for(auto [v, w]: adj[u]) {
                auto new_dist = getDist(u, v, w);
                auto old_dist = getDist(v);
                if(new_dist <= old_dist && new_dist < B) {
                    updateDist(u, v, w);
                    heap.push(new_dist);
                }
            }
        }
        if(complete.size() <= k) return {B, complete};
 
        uniqueDistT nB = getDist(complete.back());
        complete.pop_back();

        return {nB, complete};
    }
 
    std::vector<batchPQ<uniqueDistT>> Ds;
    std::vector<short int> last_complete_lvl;
    std::pair<uniqueDistT, std::vector<int>> bmsspRec(short int l, uniqueDistT B, const std::vector<int> &S) { // Algorithm 3
        if(l == 0) return baseCase(B, S[0]);
        
        auto [P, bellman_vis] = findPivots(B, S);
 
        const long long batch_size = (1ll << ((l - 1) * t));
        auto &D = Ds[l - 1];
        D.initialize(batch_size, B);
        for(int p: P) D.insert(getDist(p));
 
        uniqueDistT last_complete_B = B;
        for(int p: P) last_complete_B = std::min(last_complete_B, getDist(p));
 
        std::vector<int> complete;
        const long long quota = k * (1ll << (l * t));
        complete.reserve(quota + bellman_vis.size());
        while(complete.size() < quota && D.size()) {
            auto [trying_B, miniS] = D.pull();
            // all with dist < trying_B, can be reached by miniS <= req 2, alg 3
            auto [complete_B, nw_complete] = bmsspRec(l - 1, trying_B, miniS);
            
            // all new complete_B are greater than the old ones <= point 6, page 10
            // assert(last_complete_B < complete_B);
 
            complete.insert(complete.end(), nw_complete.begin(), nw_complete.end());
            // point 6, page 10 => complete does not intersect with nw_complete
            // assert(isUnique(complete));
 
            std::vector<uniqueDistT> can_prepend;
            can_prepend.reserve(nw_complete.size() * 5 + miniS.size());
            for(int u: nw_complete) {
                D.erase(u); // priority queue fix
                last_complete_lvl[u] = l;
                for(auto [v, w]: adj[u]) {
                    auto new_dist = getDist(u, v, w);
                    if(new_dist <= getDist(v)) {
                        updateDist(u, v, w);
                        if(trying_B <= new_dist && new_dist < B) {
                            D.insert(new_dist); // d[v] can be greater equal than std::min(D), occur 1x per vertex
                        } else if(complete_B <= new_dist && new_dist < trying_B) {
                            can_prepend.emplace_back(new_dist); // d[v] is less than all in D, can occur 1x at each level per vertex
                        }
                    }
                }
            }
            for(int x: miniS) {
                if(complete_B <= getDist(x)) can_prepend.emplace_back(getDist(x));
                // second condition is not necessary
            }
            // can_prepend is not necessarily all unique
            D.batchPrepend(can_prepend);
 
            last_complete_B = complete_B;
        }
        uniqueDistT retB;
        if(D.size() == 0) retB = B;     // successful
        else retB = last_complete_B;    // partial
 
        for(int x: bellman_vis) if(last_complete_lvl[x] != l && getDist(x) < retB) {
            complete.push_back(x); // this get the completed vertices from bellman-ford, it has P in it as well
        }
        // get only the ones not in complete already, for it to become disjoint
        return {retB, complete};
    }
};
}