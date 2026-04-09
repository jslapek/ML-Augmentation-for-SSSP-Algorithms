#include "bmssp.hpp"
#include "utils.hpp"


namespace spp_timed_cpred {

inline bool pred_c = false;
inline bool pred_pivots = false;
inline bool local_search = true; // breaks correctness when false if BF_steps != k
inline int BF_steps = 1;

void set_pred_c(bool x){
    pred_c = x;
}

void set_pred_pivots(bool x){
    pred_pivots = x;
}

template<typename uniqueDistT>
class batchPQ { // batch priority queue
    template<typename K, typename V>
    using hash_map = ankerl::unordered_dense::map<K, V>;
    using elementT = std::pair<int,uniqueDistT>;
    
    struct CompareUB {
        template <typename It>
        bool operator()(const std::pair<uniqueDistT, It>& a, const std::pair<uniqueDistT, It>& b) const {
            if (a.first != b.first) return a.first < b.first;
            return  addressof(*a.second) < addressof(*b.second);
        }
    };
    
    typename std::list<std::list<elementT>>::iterator it_min;
    
    std::list<std::list<elementT>> D0,D1;
    std::set<std::pair<uniqueDistT,typename std::list<std::list<elementT>>::iterator>,CompareUB> UBs;
    
    int M,size_;
    uniqueDistT B;
    
    hash_map<int, uniqueDistT> actual_value;
    hash_map<int, std::pair<typename std::list<std::list<elementT>>::iterator, typename std::list<elementT>::iterator>> where_is0, where_is1;
    
public:
    double snip_split = 0.0;
    double snip_lower_bound = 0.0;
    double snip_block_insertion = 0.0;
    double snip_membership_check = 0.0;
    double snip_deletion = 0.0;
    bool time_delete = false;

    batchPQ(int n): actual_value(n), where_is0(n), where_is1(n){} // O(n)

    void initialize(int M_, uniqueDistT B_) { // O(1)
        M = M_; B = B_;
        D0 = {};
        D1 = {std::list<elementT>()};
        UBs = {make_pair(B_,D1.begin())};
        size_ = 0;
        snip_split = 0.0;
        snip_lower_bound = 0.0;
        snip_block_insertion = 0.0;
        snip_membership_check = 0.0;
        snip_deletion = 0.0;
        time_delete = false;

        actual_value.clear();
        where_is0.clear(); where_is1.clear();
    }

    int size(){
        return size_;
    }
    
    void insert(uniqueDistT x){ // O(lg(Block Numbers))         
        uniqueDistT b = x;
        int a = get<2>(b);
    
        // checking if exists
        timerT timer;
        auto it_exist = actual_value.find(a);
        int exist = (it_exist != actual_value.end()); 
        timer.stop();
        snip_membership_check += timer.elapsed_ms();
    
        if(exist && it_exist->second > b){
            time_delete = true;
            delete_(x);
        }else if(exist){
            return;
        }
        
        // Searching for the first block with UB which is > 
        timer.start();
        auto it_UB_block = UBs.lower_bound({b,it_min});
        auto [ub,it_block] = (*it_UB_block);
        timer.stop();
        snip_lower_bound += timer.elapsed_ms();
        
        // Inserting key/value (a,b)
        timer.start();
        auto it = it_block->insert(it_block->end(),{a,b});
        where_is1[a] = {it_block, it};
        actual_value[a] = b;
        timer.stop();
        snip_block_insertion += timer.elapsed_ms();
    
        size_++;
    
        // Checking if exceeds the sixe limit M
        if((*it_block).size() > M){
            split(it_block);
        }
    }
    
    void batchPrepend(const std::vector<uniqueDistT> &v){
        std::list<elementT> l;
        for(auto x: v){
            l.push_back({get<2>(x),x});
        }
        batchPrepend(l);
    }

    std::pair<uniqueDistT, std::vector<int>> pull(){ // O(M)
        time_delete = false;
        std::vector<elementT> s0,s1;
        s0.reserve(2 * M); s1.reserve(M);
    
        auto it_block = D0.begin();
        while(it_block != D0.end() && s0.size() <= M){ // O(M)   
            for (const auto& x : *it_block) s0.push_back(x);
            it_block++;
        }
    
        it_block = D1.begin();
        while(it_block != D1.end() && s1.size() <= M){   //O(M)
            for (const auto& x : *it_block) s1.push_back(x);
            it_block++;
        }
    
        if(s1.size() + s0.size() <= M){
            std::vector<int> ret;
            ret.reserve(s1.size()+s0.size());
            for(auto [a,b] : s0) {
                ret.push_back(a);
                delete_({b});
            }
            for(auto [a,b] : s1){
                ret.push_back(a);
                delete_({b});
            } 
            
            return {B, ret};
        }else{  
            std::vector<elementT> &l = s0;
            l.insert(l.end(), s1.begin(), s1.end());

            uniqueDistT med = selectKth(l, M);
            std::vector<int> ret;
            ret.reserve(M);
            for(auto [a,b]: l){
                if(b < med) {
                    ret.push_back(a);
                    delete_({b});
                }
            }
            return {med,ret};
        }
    }
    inline void erase(int key) {
        if(actual_value.find(key) != actual_value.end()) {
            time_delete = true;
            delete_({-1, -1, key, -1});
        }
    }
    
private:
    void delete_(uniqueDistT x){    
        timerT timer;
        int a = get<2>(x);
        uniqueDistT b = actual_value[a];
        
        auto it_w = where_is1.find(a);
        if((it_w != where_is1.end())){
            auto [it_block,it] = it_w->second;
            
            (*it_block).erase(it);
            where_is1.erase(a);
    
            if((*it_block).size() == 0){
                timer.stop();
                if (time_delete) snip_deletion += timer.elapsed_ms();
                timer.start();
                auto it_UB_block = UBs.lower_bound({b,it_block});  
                timer.stop();
                if (time_delete) snip_lower_bound += timer.elapsed_ms();

                timer.start();
                if((*it_UB_block).first != B){
                    UBs.erase(it_UB_block);
                    D1.erase(it_block);
                }
            }
        }else{
            auto [it_block,it] = where_is0[a];
            (*it_block).erase(it);
            where_is0.erase(a);
            if((*it_block).size() == 0) D0.erase(it_block); 
        }
    
        actual_value.erase(a);
        size_--;
        timer.stop();
        if (time_delete) snip_deletion += timer.elapsed_ms();
    }
    
    uniqueDistT selectKth(std::vector<elementT> &v, int k) {
        const auto comparator = [](const auto &a, const auto &b){
            return a.second < b.second;
        };
        miniselect::floyd_rivest_select(v.begin(), v.begin() + k, v.end(), comparator);
        return v[k].second;
    }

        
    void split(std::list<std::list<elementT>>::iterator it_block){ // O(M) + O(lg(Block Numbers))
        timerT timer;
        int sz = (*it_block).size();
        
        std::vector<elementT> v((*it_block).begin() , (*it_block).end());
        uniqueDistT med = selectKth(v,(sz/2)); // O(M)
        
        auto pos = it_block;
        pos++;

        auto new_block = D1.insert(pos,std::list<elementT>());
        auto it = (*it_block).begin();
    
        while(it != (*it_block).end()){ // O(M)
            if((*it).second >= med){
                (*new_block).push_back(move(*it));
                auto it_new = (*new_block).end(); it_new--;
                where_is1[(*it).first] = {new_block, it_new};
    
                it = (*it_block).erase(it);
            }else{
                it++;
            }
        }
    

        // Updating UBs   
        // O(lg(Block Numbers))
        uniqueDistT UB1 = {get<0>(med),get<1>(med),get<2>(med),get<3>(med)-1};
        timer.stop();
        snip_split += timer.elapsed_ms();

        timer.start();
        auto it_lb = UBs.lower_bound({UB1,it_min});
        auto [UB2,aux] = (*it_lb);
        timer.stop();
        snip_lower_bound += timer.elapsed_ms();

        
        timer.start();
        UBs.insert({UB1,it_block});
        UBs.insert({UB2,new_block});
        
        UBs.erase(it_lb);
        timer.stop();
        snip_split += timer.elapsed_ms();

    }
    
    void batchPrepend(const std::list<elementT> &l) { // O(|l| log(|l|/M) ) 
        int sz = l.size();
        
        if(sz == 0) return;
        if(sz <= M){
    
            D0.push_front(std::list<elementT>());
            auto new_block = D0.begin();
            
            for(auto &x : l){ 
                auto it = actual_value.find(x.first);
                int exist = (it != actual_value.end()); 
    
                if(exist && it->second > x.second){
                    time_delete = false;
                    delete_(x.second);
                }else if(exist){
                    continue;
                }
    
                (*new_block).push_back(x);
                auto it_new = (*new_block).end(); it_new--;
                where_is0[x.first] = {new_block, it_new};
                actual_value[x.first] = x.second;
                size_++;
            }
            if(new_block->size() == 0) D0.erase(new_block);
            return;
        }

        std::vector<elementT> v(l.begin(), l.end());
        uniqueDistT med = selectKth(v, sz/2);
    
        std::list<elementT> less,great;
        for(auto [a,b]: l){
            if(b < med){
                less.push_back({a,b});
            }else if(b > med){
                great.push_back({a,b});
            }
        }
        
        great.push_back({get<2>(med),med});

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

public:
    Stats stats;
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
        timerT timer;
        fill(d.begin(), d.end(), oo);
        fill(last_complete_lvl.begin(), last_complete_lvl.end(), -1);
        fill(pivot_vis.begin(), pivot_vis.end(), -1);
        fill(path_sz.begin(), path_sz.end(), 0);
        fill(root.begin(), root.end(), -1);
        fill(treesz.begin(), treesz.end(), 0);
        for(int i = 0; i < (int)pred.size(); i++) pred[i] = i;
        
        s = toAnyCustomNode(s);
        d[s] = 0;
        path_sz[s] = 0;
        pred[s] = s;
        
        const int l = ceil(log2(adj.size()) / t);
        const uniqueDistT inf_dist = {oo, 0, 0, 0};
        timer.stop();
        stats.time_full -= timer.elapsed_ms();
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



    struct FindPivotsUndo {
        struct DistState {
            int u;
            wT old_d;
            int old_pred;
            int old_path_sz;
        };

        std::vector<DistState> dist_changes;
        std::vector<std::pair<int, int>> root_changes;
        std::vector<std::pair<int, short int>> treesz_changes;
        std::vector<std::pair<int, int>> pivot_vis_changes;

        int old_counter_pivot = 0;

        // optional but recommended for exact rollback
        double old_snip_relaxation = 0.0;
        double old_snip_tree_construction = 0.0;

        void rollback(bmssp &self) {
            for (auto it = dist_changes.rbegin(); it != dist_changes.rend(); ++it) {
                self.d[it->u] = it->old_d;
                self.pred[it->u] = it->old_pred;
                self.path_sz[it->u] = it->old_path_sz;
            }
            for (auto it = root_changes.rbegin(); it != root_changes.rend(); ++it) {
                self.root[it->first] = it->second;
            }
            for (auto it = treesz_changes.rbegin(); it != treesz_changes.rend(); ++it) {
                self.treesz[it->first] = it->second;
            }
            for (auto it = pivot_vis_changes.rbegin(); it != pivot_vis_changes.rend(); ++it) {
                self.pivot_vis[it->first] = it->second;
            }

            self.counter_pivot = old_counter_pivot;
            self.stats.snip_relaxation = old_snip_relaxation;
            self.stats.snip_tree_construction = old_snip_tree_construction;
        }
    };

    std::vector<int> root;
    std::vector<short int> treesz;

    int counter_pivot = 0;
    std::vector<int> pivot_vis;
    
    std::pair<std::vector<int>, std::vector<int>> findPivots(
        uniqueDistT B, const std::vector<int> &S, FindPivotsUndo *undo = nullptr
    ) { // Algorithm 1
        timerT timer;

        ///////// Logging for undo
        if (undo) {
            undo->old_counter_pivot = counter_pivot;
            undo->old_snip_relaxation = stats.snip_relaxation;
            undo->old_snip_tree_construction = stats.snip_tree_construction;
        }

        std::unordered_set<int> seen_dist;
        std::unordered_set<int> seen_root;
        std::unordered_set<int> seen_treesz;
        std::unordered_set<int> seen_pivot;

        auto log_dist = [&](int u) {
            if (!undo) return;
            if (seen_dist.insert(u).second) {
                undo->dist_changes.push_back({u, d[u], pred[u], path_sz[u]});
            }
        };

        auto log_root = [&](int u) {
            if (!undo) return;
            if (seen_root.insert(u).second) {
                undo->root_changes.push_back({u, root[u]});
            }
        };

        auto log_treesz = [&](int u) {
            if (!undo) return;
            if (seen_treesz.insert(u).second) {
                undo->treesz_changes.push_back({u, treesz[u]});
            }
        };

        auto log_pivot = [&](int u) {
            if (!undo) return;
            if (seen_pivot.insert(u).second) {
                undo->pivot_vis_changes.push_back({u, pivot_vis[u]});
            }
        };
        timer.stop();
        stats.time_full -= timer.elapsed_ms();
        /////////////

        counter_pivot++;

        std::vector<int> vis;
        vis.reserve(2 * k * S.size());

        for (int x : S) {
            vis.push_back(x);
            log_pivot(x);
            pivot_vis[x] = counter_pivot;
        }

        std::vector<int> active = S;
        for (int x : S) {
            log_root(x);
            root[x] = x;

            log_treesz(x);
            treesz[x] = 0;
        }

        for (int i = 1; i <= k; i++) {
            std::vector<int> nw_active;
            nw_active.reserve(active.size() * 4);

            for (int u : active) {
                for (auto [v, w] : adj[u]) {
                    if (getDist(u, v, w) <= getDist(v)) {
                        log_dist(v);

                        timer.start();
                        updateDist(u, v, w);
                        if (getDist(v) < B) {
                            log_root(v);
                            root[v] = root[u];
                            nw_active.push_back(v);
                        }
                        timer.stop();
                        stats.snip_relaxation += timer.elapsed_ms();
                    }
                }
            }

            for (const auto &x : nw_active) {
                if (pivot_vis[x] != counter_pivot) {
                    log_pivot(x);
                    pivot_vis[x] = counter_pivot;
                    vis.push_back(x);
                }
            }

            if (vis.size() > k * S.size()) {
                return {S, vis};
            }
            active = move(nw_active);
        }

        timer.start();
        std::vector<int> P;
        P.reserve(vis.size() / k);

        for (int u : vis) {
            log_treesz(root[u]);
            treesz[root[u]]++;
        }
        for (int u : S) {
            if (treesz[u] >= k) P.push_back(u);
        }

        timer.stop();
        if (pred_pivots && !(pred_c && P == S)) {
            stats.time_full -= timer.elapsed_ms();
        } 

        return {P, vis};
    }

    std::pair<std::vector<int>, std::vector<int>> repairOrPromote(
        uniqueDistT B,
        const std::vector<int> &S,
        const std::vector<int> &P_hat
    ) {
        struct OwnedState {
            uniqueDistT dist;
            int owner;
            OwnedState() = default;
            OwnedState(uniqueDistT d, int o): dist(d), owner(o) {}
            bool operator>(const OwnedState &other) const {
                if (dist != other.dist) return dist > other.dist;
                return owner > other.owner;
            }
        };

        const int N = (int)adj.size();
        const uniqueDistT inf_dist = {oo, 0, 0, 0};

        std::vector<char> is_source(N, false), in_pivot(N, false), is_dropped(N, false);
        std::vector<int> cnt(N, 0);
        std::vector<uniqueDistT> best_dist(N, inf_dist);
        std::vector<int> best_owner(N, -1);
        std::vector<char> settled(N, false);

        std::unordered_map<int, std::vector<int>> buckets;
        buckets.reserve(S.size() * 2 + 1);

        std::vector<int> P = P_hat;
        P.reserve(S.size());

        for (int s : S) is_source[s] = true;
        for (int p : P_hat) in_pivot[p] = true;
        for (int s : S) {
            if (!in_pivot[s]) {
                is_dropped[s] = true;
                cnt[s] = 1; // count the complete root itself
                buckets.emplace(s, std::vector<int>());
                buckets[s].reserve(std::max(0, k - 1));
            }
        }

        std::priority_queue<OwnedState, std::vector<OwnedState>, std::greater<OwnedState>> heap;

        // Seed with all roots so that ownership is computed exactly under global competition.
        for (int s : S) {
            auto ds = getDist(s);
            best_dist[s] = ds;
            best_owner[s] = s;
            heap.emplace(ds, s);
        }

        while (!heap.empty()) {
            auto cur = heap.top();
            heap.pop();

            const uniqueDistT du = cur.dist;
            const int owner = cur.owner;
            const int u = get<2>(du);

            if (B <= du) break;
            if (best_owner[u] != owner || best_dist[u] != du) continue;
            if (settled[u]) continue;
            settled[u] = true;

            // Record exact vertices owned by currently dropped roots, but do not write to global
            // state yet. If a root is later promoted, its bucket is simply discarded.
            if (is_dropped[owner] && !in_pivot[owner] && u != owner) {
                auto &bucket = buckets[owner];
                bucket.push_back(u);
                if (++cnt[owner] >= k) {
                    in_pivot[owner] = true;
                    P.push_back(owner);
                    bucket.clear();
                }
            }

            // Continue expanding from all owners (including promoted ones) so that ownership of
            // every explored vertex is still computed under exact global competition.
            for (auto [v, w] : adj[u]) {
                if (is_source[v]) continue; // roots of S act as absorbing boundaries

                uniqueDistT cand{get<0>(du) + w, get<1>(du) + 1, v, u};
                if (!(cand < B)) continue;

                if (best_owner[v] == -1 || cand < best_dist[v] ||
                    (cand == best_dist[v] && owner < best_owner[v])) {
                    best_dist[v] = cand;
                    best_owner[v] = owner;
                    heap.emplace(cand, owner);
                }
            }
        }

        // Commit only vertices of dropped roots that were fully repaired (i.e. not promoted).
        std::vector<int> C;
        C.reserve(k * std::max<size_t>(1, S.size()));
        for (int s : S) {
            if (!is_dropped[s] || in_pivot[s]) continue;
            auto it = buckets.find(s);
            if (it == buckets.end()) continue;
            for (int u : it->second) {
                pred[u] = get<3>(best_dist[u]);
                d[u] = get<0>(best_dist[u]);
                path_sz[u] = get<1>(best_dist[u]);
                C.push_back(u);
            }
        }

        return {P, C};
    }

    std::pair<std::vector<int>, std::vector<int>> p_findPivots(
        uniqueDistT B, const std::vector<int> &S, const std::vector<int> &pred_P
    ) {
        counter_pivot++;

        // Prefix exploration is used only as a cheap cutoff heuristic. It must not mutate the
        // global distance state, since only the exact repair phase is allowed to certify vertices.
        ankerl::unordered_dense::map<int, uniqueDistT> local_dist;
        local_dist.reserve(4 * std::max<size_t>(1, k * S.size()));

        std::vector<int> vis;
        vis.reserve(2 * k * S.size());
        for (int x : S) {
            vis.push_back(x);
            pivot_vis[x] = counter_pivot;
        }

        std::vector<int> active = S;
        for (int i = 1; i <= BF_steps; i++) {
            std::vector<int> nw_active;
            nw_active.reserve(active.size() * 4);

            for (int u : active) {
                uniqueDistT du = getDist(u);
                auto it_u = local_dist.find(u);
                if (it_u != local_dist.end()) du = it_u->second;

                for (auto [v, w] : adj[u]) {
                    uniqueDistT cand{get<0>(du) + w, get<1>(du) + 1, v, u};
                    uniqueDistT curv = getDist(v);
                    auto it_v = local_dist.find(v);
                    if (it_v != local_dist.end()) curv = it_v->second;

                    if (cand <= curv) {
                        local_dist[v] = cand;
                        if (cand < B) {
                            nw_active.push_back(v);
                        }
                    }
                }
            }

            for (int x : nw_active) {
                if (pivot_vis[x] != counter_pivot) {
                    pivot_vis[x] = counter_pivot;
                    vis.push_back(x);
                }
            }

            if (vis.size() > static_cast<size_t>(k) * S.size()) {
                return {S, {}};
            }

            active = std::move(nw_active);
        }

        auto [P_final, C] = repairOrPromote(B, S, pred_P);
        return {P_final, C};
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
        if(l == 0) {
            timerT timer;
            auto x = baseCase(B, S[0]);
            timer.stop();
            stats.time_base_case += timer.elapsed_ms();
            return x;
        }

        timerT timer;

        std::vector<int> P;
        std::vector<int> bellman_vis;

        {
            FindPivotsUndo undo;
            auto res = findPivots(B, S, &undo);
            P = std::move(res.first);
            bellman_vis = std::move(res.second);

            if (local_search || (pred_c && P == S)) {
                timer.start();
                undo.rollback(*this);
                bellman_vis.clear(); // discard tentative exploration
                timer.stop();
                stats.time_full -= timer.elapsed_ms();
            }

            if (local_search && P != S) {
                auto [P_ls, vis_ls] = p_findPivots(B, S, P);
                if (P_ls != P) {
                    std::cout << "p_findPivots incorrect";
                }
                // P = std::move(P_ls);
                bellman_vis = std::move(vis_ls);
            }
        } 

        const long long batch_size = (1ll << ((l - 1) * t));
        auto &D = Ds[l - 1];
        D.initialize(batch_size, B);
        
        timer.start();
        for(int p: P) D.insert(getDist(p));
        timer.stop();
        stats.time_D_op += timer.elapsed_ms();

        uniqueDistT last_complete_B = B;
        for(int p: P) last_complete_B = std::min(last_complete_B, getDist(p));

        std::vector<int> complete;
        const long long quota = k * (1ll << (l * t));
        complete.reserve(quota + bellman_vis.size());
        while(complete.size() < quota && D.size()) {
            auto [trying_B, miniS] = D.pull();
            auto [complete_B, nw_complete] = bmsspRec(l - 1, trying_B, miniS);

            complete.insert(complete.end(), nw_complete.begin(), nw_complete.end());

            std::vector<uniqueDistT> can_prepend;
            can_prepend.reserve(nw_complete.size() * 5 + miniS.size());
            for(int u: nw_complete) {
                timer.start();
                D.erase(u);
                timer.stop();
                stats.time_D_op += timer.elapsed_ms();

                last_complete_lvl[u] = l;
                for(auto [v, w]: adj[u]) {
                    auto new_dist = getDist(u, v, w);
                    if(new_dist <= getDist(v)) {
                        updateDist(u, v, w);
                        if(trying_B <= new_dist && new_dist < B) {
                            timer.start();
                            D.insert(new_dist);
                            timer.stop();
                            stats.time_D_op += timer.elapsed_ms();
                        } else if(complete_B <= new_dist && new_dist < trying_B) {
                            can_prepend.emplace_back(new_dist);
                        }
                    }
                }
            }
            for(int x: miniS) {
                if(complete_B <= getDist(x)) can_prepend.emplace_back(getDist(x));
            }

            timer.start();
            D.batchPrepend(can_prepend);
            timer.stop();
            stats.time_batch_prepend += timer.elapsed_ms();

            last_complete_B = complete_B;
        }

        uniqueDistT retB;
        if(D.size() == 0) retB = B;
        else retB = last_complete_B;

        for(int x: bellman_vis) if(last_complete_lvl[x] != l && getDist(x) < retB) {
            complete.push_back(x);
        }

        stats.snip_lower_bound += D.snip_lower_bound;
        stats.snip_split += D.snip_split;
        stats.snip_block_insertion += D.snip_block_insertion;
        stats.snip_membership_check += D.snip_membership_check;
        stats.snip_deletion += D.snip_deletion;
        
        return {retB, complete};
    }


};
}