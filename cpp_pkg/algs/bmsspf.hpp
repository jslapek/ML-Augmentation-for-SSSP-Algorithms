#include "bmssp.hpp"
#include "utils.hpp"
#include "bmssp_lapq.hpp"
#include "../use_pscase_model.hpp"
#include "../use_countmin_model.hpp"

namespace spp_bmsspf {

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

    batchPQ(int n): actual_value(n), where_is0(n), where_is1(n){} // O(n)

    void initialize(int M_, uniqueDistT B_) { // O(1)
        M = M_; B = B_;
        D0 = {};
        D1 = {std::list<elementT>()};
        UBs = {make_pair(B_,D1.begin())};
        size_ = 0;

        actual_value.clear();
        where_is0.clear(); where_is1.clear();
    }

    int size(){
        return size_;
    }
    
    void insert(uniqueDistT x){ // O(lg(Block Numbers))         
        uniqueDistT b = x;
        int a = get<2>(b);
    
        auto it_exist = actual_value.find(a);
        int exist = (it_exist != actual_value.end());
    
        if(exist && it_exist->second > b){
            delete_(x);
        }else if(exist){
            return;
        }
        
        auto it_UB_block = UBs.lower_bound({b,it_min});
        auto [ub,it_block] = (*it_UB_block);
        
        auto it = it_block->insert(it_block->end(),{a,b});
        where_is1[a] = {it_block, it};
        actual_value[a] = b;
    
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
            delete_({-1, -1, key, -1});
        }
    }
    
private:

    void delete_(uniqueDistT x){
        int a = get<2>(x);
        uniqueDistT b = actual_value[a];

        auto it_w = where_is1.find(a);
        if((it_w != where_is1.end())){
            auto [it_block,it] = it_w->second;

            (*it_block).erase(it);
            where_is1.erase(a);

            if((*it_block).size() == 0){
                auto it_UB_block = UBs.lower_bound({b,it_block});
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
    }

    uniqueDistT selectKth(std::vector<elementT> &v, int k) {
        const auto comparator = [](const auto &a, const auto &b){
            return a.second < b.second;
        };
        miniselect::floyd_rivest_select(v.begin(), v.begin() + k, v.end(), comparator);
        return v[k].second;
    }

        

    void split(std::list<std::list<elementT>>::iterator it_block){ // O(M) + O(lg(Block Numbers))
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

        uniqueDistT UB1 = {get<0>(med),get<1>(med),get<2>(med),get<3>(med)-1};
        auto it_lb = UBs.lower_bound({UB1,it_min});
        auto [UB2,aux] = (*it_lb);

        UBs.insert({UB1,it_block});
        UBs.insert({UB2,new_block});

        UBs.erase(it_lb);
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
    PSEqSModel ps_model;
    PSEqSState ps_state;
    bool use_ps_predictor = true;

    CountMinModel cm_model;
    CountMinState cm_state;
    bool use_countmin_predictor = false;

    enum class queue_backend_t { bpq, lapq };
    enum class pivot_search_backend_t { disabled, dedup, npf, hybrid, ob };
    queue_backend_t queue_backend = queue_backend_t::bpq;
    pivot_search_backend_t pivot_search_backend = pivot_search_backend_t::disabled;
    int bf_prefix_steps = 0;
    std::string predictor_graph_name_current = "randomD";
    std::string countmin_mode_current = "false";

    static queue_backend_t parse_queue_backend(const std::string& mode) {
        if(mode == "bpq") return queue_backend_t::bpq;
        if(mode == "lapq") return queue_backend_t::lapq;
        throw std::invalid_argument("queue_mode must be \"bpq\" or \"lapq\"");
    }

    static pivot_search_backend_t parse_pivot_search_backend(const std::string& mode) {
        if(mode == "false") return pivot_search_backend_t::disabled;
        if(mode == "dedup") return pivot_search_backend_t::dedup;
        if(mode == "npf") return pivot_search_backend_t::npf;
        if(mode == "hybrid") return pivot_search_backend_t::hybrid;
        if(mode == "ob") return pivot_search_backend_t::ob;
        throw std::invalid_argument("pivot_search_mode must be one of \"false\", \"dedup\", \"npf\", \"hybrid\", or \"ob\"");
    }

public:
    Stats stats;
    const wT oo = std::numeric_limits<wT>::max() / 10;
    bmssp(
        int n_,
        const std::string& predictor_graph_name = "randomD",
        const std::string& predictor_mode = "offline",
        const std::string& queue_mode = "bpq",
        const std::string& pivot_search_mode = "false",
        const std::string& countmin_mode = "false",
        int BF_steps = 0
    )
        : n(n_),
          ps_model(get_P_eq_S_model(predictor_graph_name, (predictor_mode == "false" || predictor_mode == "none") ? "blank" : predictor_mode)),
          use_ps_predictor(predictor_mode != "false" && predictor_mode != "none"),
          cm_model(get_countmin_model(predictor_graph_name, countmin_mode == "false" ? "blank" : countmin_mode)),
          use_countmin_predictor(countmin_mode != "false"),
          queue_backend(parse_queue_backend(queue_mode)),
          pivot_search_backend(parse_pivot_search_backend(pivot_search_mode)),
          bf_prefix_steps(BF_steps),
          predictor_graph_name_current(predictor_graph_name),
          countmin_mode_current(countmin_mode) {
        ori_adj.assign(n, {});
    }
    bmssp(
        const auto &adj,
        const std::string& predictor_graph_name = "randomD",
        const std::string& predictor_mode = "offline",
        const std::string& queue_mode = "bpq",
        const std::string& pivot_search_mode = "false",
        const std::string& countmin_mode = "false",
        int BF_steps = 0
    )
        : n(adj.size()),
          ori_adj(adj),
          ps_model(get_P_eq_S_model(predictor_graph_name, (predictor_mode == "false" || predictor_mode == "none") ? "blank" : predictor_mode)),
          use_ps_predictor(predictor_mode != "false" && predictor_mode != "none"),
          cm_model(get_countmin_model(predictor_graph_name, countmin_mode == "false" ? "blank" : countmin_mode)),
          use_countmin_predictor(countmin_mode != "false"),
          queue_backend(parse_queue_backend(queue_mode)),
          pivot_search_backend(parse_pivot_search_backend(pivot_search_mode)),
          bf_prefix_steps(BF_steps),
          predictor_graph_name_current(predictor_graph_name),
          countmin_mode_current(countmin_mode) {}

    void set_predictor_graph(
        const std::string& graph_name,
        const std::string& predictor_mode = "offline",
        const std::string& countmin_mode = "false"
    ) {
        predictor_graph_name_current = graph_name;
        countmin_mode_current = countmin_mode;
        use_ps_predictor = (predictor_mode != "false" && predictor_mode != "none");
        ps_model = get_P_eq_S_model(graph_name, use_ps_predictor ? predictor_mode : "blank");
        ps_state = PSEqSState{};

        use_countmin_predictor = (countmin_mode != "false");
        cm_model = get_countmin_model(graph_name, use_countmin_predictor ? countmin_mode : "blank");
        cm_state = CountMinState{};
    }

    void set_queue_mode(const std::string& queue_mode = "bpq") {
        queue_backend = parse_queue_backend(queue_mode);
    }

    void set_pivot_search_mode(
        const std::string& pivot_search_mode = "false",
        const std::string& countmin_mode = "false",
        int BF_steps = 0
    ) {
        pivot_search_backend = parse_pivot_search_backend(pivot_search_mode);
        countmin_mode_current = countmin_mode;
        use_countmin_predictor = (countmin_mode != "false");
        cm_model = get_countmin_model(predictor_graph_name_current, use_countmin_predictor ? countmin_mode : "blank");
        cm_state = CountMinState{};
        bf_prefix_steps = BF_steps;
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
        round_seen_stamp.assign(adj.size(), 0);
        pivot_member_stamp.assign(adj.size(), 0);
        cand_dist_val.resize(adj.size());
        cand_dist_stamp.assign(adj.size(), 0);
        cand_owner_val.assign(adj.size(), 0);
        owner_count_val.assign(adj.size(), 0);
        owner_count_stamp.assign(adj.size(), 0);
        owner_bucket_count.assign(adj.size(), 0);
        owner_bucket_start.assign(adj.size(), 0);
        owner_bucket_pos.assign(adj.size(), 0);
        owner_bucket_min_val.resize(adj.size());
        owner_bucket_stamp.assign(adj.size(), 0);
        scratch_nonpivot.clear();
        scratch_pivot.clear();
        scratch_owner_nonpivot.clear();
        scratch_owner_pivot.clear();
        scratch_bucket_vertices.clear();
        scratch_touched_targets.clear();
        k = floor(pow(log2(adj.size()), 1.0 / 3.0));
        t = floor(pow(log2(adj.size()), 2.0 / 3.0));
        l = ceil(log2(adj.size()) / t);
        Ds.clear();
        Ds.reserve(l);
        for(int i = 0; i < l; i++) {
            if(queue_backend == queue_backend_t::lapq) Ds.push_back(std::make_unique<spp_lapq::lapq_batchPQ<uniqueDistT>>(static_cast<int>(adj.size())));
            else Ds.push_back(std::make_unique<spp_lapq::bpq_batchPQ<uniqueDistT>>(static_cast<int>(adj.size())));
        }
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

    std::vector<uniqueDistT> cand_dist_val;
    std::vector<int> cand_dist_stamp;
    std::vector<int> cand_owner_val;
    std::vector<int> owner_count_val;
    std::vector<int> owner_count_stamp;
    std::vector<int> round_seen_stamp;
    std::vector<int> pivot_member_stamp;
    std::vector<int> owner_bucket_count;
    std::vector<int> owner_bucket_start;
    std::vector<int> owner_bucket_pos;
    std::vector<uniqueDistT> owner_bucket_min_val;
    std::vector<int> owner_bucket_stamp;
    std::vector<int> scratch_touched_targets;
    std::vector<int> scratch_nonpivot;
    std::vector<int> scratch_pivot;
    std::vector<int> scratch_owner_nonpivot;
    std::vector<int> scratch_owner_pivot;
    std::vector<int> scratch_bucket_vertices;

    int cand_token = 0;
    int owner_count_token = 0;
    int round_seen_token = 0;
    int pivot_member_token = 0;
    int owner_bucket_token = 0;

    inline void nextStamp(int &token, std::vector<int> &stamp) {
        ++token;
        if (token == std::numeric_limits<int>::max()) {
            std::fill(stamp.begin(), stamp.end(), 0);
            token = 1;
        }
    }

    inline void nextCandEpoch() { nextStamp(cand_token, cand_dist_stamp); }
    inline void nextOwnerCountEpoch() { nextStamp(owner_count_token, owner_count_stamp); }
    inline void nextOwnerBucketEpoch() { nextStamp(owner_bucket_token, owner_bucket_stamp); }

    int counter_pivot = 0;
    std::vector<int> pivot_vis;

    std::vector<int> relaxRoundPrefixActual(
        const std::vector<int> &Q,
        uniqueDistT B,
        std::vector<int> &vis
    ) {
        nextStamp(round_seen_token, round_seen_stamp);
        std::vector<int> R;
        R.reserve(Q.size() * 2 + 4);
        for (int u : Q) {
            for (const auto &[v, w] : adj[u]) {
                if (getDist(u, v, w) <= getDist(v)) {
                    updateDist(u, v, w);
                    if (getDist(v) < B) {
                        root[v] = root[u];
                        if (round_seen_stamp[v] != round_seen_token) {
                            round_seen_stamp[v] = round_seen_token;
                            R.push_back(v);
                        }
                        if (pivot_vis[v] != counter_pivot) {
                            pivot_vis[v] = counter_pivot;
                            vis.push_back(v);
                        }
                    }
                }
            }
        }
        return R;
    }

    std::vector<int> scheduleRoundDedupActual(
        const std::vector<int> &Q,
        uniqueDistT B,
        std::vector<int> &vis
    ) {
        nextCandEpoch();
        scratch_touched_targets.clear();
        scratch_touched_targets.reserve(Q.size() * 2 + 4);

        for (int u : Q) {
            const int owner_u = root[u];
            for (const auto &[v, w] : adj[u]) {
                const uniqueDistT cand{d[u] + w, path_sz[u] + 1, v, u};
                if (cand_dist_stamp[v] != cand_token) {
                    cand_dist_stamp[v] = cand_token;
                    cand_dist_val[v] = getDist(v);
                    cand_owner_val[v] = root[v];
                    scratch_touched_targets.push_back(v);
                }
                if (cand < cand_dist_val[v]) {
                    cand_dist_val[v] = cand;
                    cand_owner_val[v] = owner_u;
                }
            }
        }

        std::vector<int> R;
        R.reserve(scratch_touched_targets.size());
        for (int v : scratch_touched_targets) {
            const uniqueDistT &best = cand_dist_val[v];
            if (best < getDist(v)) {
                pred[v] = get<3>(best);
                d[v] = get<0>(best);
                path_sz[v] = get<1>(best);
                root[v] = cand_owner_val[v];
                if (best < B) {
                    R.push_back(v);
                    if (pivot_vis[v] != counter_pivot) {
                        pivot_vis[v] = counter_pivot;
                        vis.push_back(v);
                    }
                }
            }
        }
        return R;
    }

    std::vector<int> scheduleRoundNPFActual(
        const std::vector<int> &Q,
        uniqueDistT B,
        std::vector<int> &vis
    ) {
        nextStamp(round_seen_token, round_seen_stamp);
        scratch_nonpivot.clear();
        scratch_pivot.clear();
        scratch_nonpivot.reserve(Q.size());
        scratch_pivot.reserve(Q.size());

        for (int u : Q) {
            const int owner_u = root[u];
            if (pivot_member_stamp[owner_u] == pivot_member_token) scratch_pivot.push_back(u);
            else scratch_nonpivot.push_back(u);
        }

        std::vector<int> R;
        R.reserve(Q.size() * 2 + 4);
        auto process_group = [&](const std::vector<int> &group) {
            for (int u : group) {
                const int owner_u = root[u];
                for (const auto &[v, w] : adj[u]) {
                    if (getDist(u, v, w) <= getDist(v)) {
                        updateDist(u, v, w);
                        if (getDist(v) < B) {
                            root[v] = owner_u;
                            if (round_seen_stamp[v] != round_seen_token) {
                                round_seen_stamp[v] = round_seen_token;
                                R.push_back(v);
                            }
                            if (pivot_vis[v] != counter_pivot) {
                                pivot_vis[v] = counter_pivot;
                                vis.push_back(v);
                            }
                        }
                    }
                }
            }
        };
        process_group(scratch_nonpivot);
        process_group(scratch_pivot);
        return R;
    }

    std::vector<int> scheduleRoundOBActual(
        const std::vector<int> &Q,
        uniqueDistT B,
        std::vector<int> &vis,
        bool hybrid_mode
    ) {
        if (Q.empty()) return {};
        constexpr size_t DIRECT_DEDUP_THRESHOLD = 32;
        constexpr int SORT_BUCKET_THRESHOLD = 8;
        if (hybrid_mode && Q.size() <= DIRECT_DEDUP_THRESHOLD) {
            return scheduleRoundDedupActual(Q, B, vis);
        }

        nextStamp(round_seen_token, round_seen_stamp);
        nextOwnerBucketEpoch();
        scratch_owner_nonpivot.clear();
        scratch_owner_pivot.clear();
        scratch_owner_nonpivot.reserve(Q.size());
        scratch_owner_pivot.reserve(Q.size());
        if (scratch_bucket_vertices.size() < Q.size()) scratch_bucket_vertices.resize(Q.size());

        for (int u : Q) {
            const int owner_u = root[u];
            const uniqueDistT du = getDist(u);
            if (owner_bucket_stamp[owner_u] != owner_bucket_token) {
                owner_bucket_stamp[owner_u] = owner_bucket_token;
                owner_bucket_count[owner_u] = 0;
                owner_bucket_min_val[owner_u] = du;
                if (pivot_member_stamp[owner_u] == pivot_member_token) scratch_owner_pivot.push_back(owner_u);
                else scratch_owner_nonpivot.push_back(owner_u);
            } else if (du < owner_bucket_min_val[owner_u]) {
                owner_bucket_min_val[owner_u] = du;
            }
            ++owner_bucket_count[owner_u];
        }

        if (hybrid_mode && scratch_owner_nonpivot.size() + scratch_owner_pivot.size() <= 1) {
            return scheduleRoundDedupActual(Q, B, vis);
        }

        auto owner_cmp = [&](int a, int b) {
            const uniqueDistT &da = owner_bucket_min_val[a];
            const uniqueDistT &db = owner_bucket_min_val[b];
            if (da != db) return da < db;
            return a < b;
        };
        if (scratch_owner_nonpivot.size() > 1) std::sort(scratch_owner_nonpivot.begin(), scratch_owner_nonpivot.end(), owner_cmp);
        if (scratch_owner_pivot.size() > 1) std::sort(scratch_owner_pivot.begin(), scratch_owner_pivot.end(), owner_cmp);

        int cur_pos = 0;
        for (int owner_u : scratch_owner_nonpivot) {
            owner_bucket_start[owner_u] = cur_pos;
            owner_bucket_pos[owner_u] = cur_pos;
            cur_pos += owner_bucket_count[owner_u];
        }
        for (int owner_u : scratch_owner_pivot) {
            owner_bucket_start[owner_u] = cur_pos;
            owner_bucket_pos[owner_u] = cur_pos;
            cur_pos += owner_bucket_count[owner_u];
        }

        for (int u : Q) {
            const int owner_u = root[u];
            scratch_bucket_vertices[owner_bucket_pos[owner_u]++] = u;
        }

        auto vertex_cmp = [&](int a, int b) {
            const uniqueDistT da = getDist(a);
            const uniqueDistT db = getDist(b);
            if (da != db) return da < db;
            return a < b;
        };
        auto maybe_sort_buckets = [&](const std::vector<int> &owners) {
            for (int owner_u : owners) {
                const int st = owner_bucket_start[owner_u];
                const int en = st + owner_bucket_count[owner_u];
                if ((!hybrid_mode && en - st > 1) || (hybrid_mode && en - st > SORT_BUCKET_THRESHOLD)) {
                    std::sort(scratch_bucket_vertices.begin() + st, scratch_bucket_vertices.begin() + en, vertex_cmp);
                }
            }
        };
        maybe_sort_buckets(scratch_owner_nonpivot);
        maybe_sort_buckets(scratch_owner_pivot);

        if (hybrid_mode) {
            nextCandEpoch();
            scratch_touched_targets.clear();
            scratch_touched_targets.reserve(Q.size() * 2 + 4);
            auto process_owner_list = [&](const std::vector<int> &owners) {
                for (int owner_u : owners) {
                    const int st = owner_bucket_start[owner_u];
                    const int en = st + owner_bucket_count[owner_u];
                    for (int idx = st; idx < en; ++idx) {
                        const int u = scratch_bucket_vertices[idx];
                        for (const auto &[v, w] : adj[u]) {
                            const uniqueDistT cand{d[u] + w, path_sz[u] + 1, v, u};
                            if (cand_dist_stamp[v] != cand_token) {
                                cand_dist_stamp[v] = cand_token;
                                cand_dist_val[v] = getDist(v);
                                cand_owner_val[v] = root[v];
                                scratch_touched_targets.push_back(v);
                            }
                            if (cand < cand_dist_val[v]) {
                                cand_dist_val[v] = cand;
                                cand_owner_val[v] = owner_u;
                            }
                        }
                    }
                }
            };
            process_owner_list(scratch_owner_nonpivot);
            process_owner_list(scratch_owner_pivot);

            std::vector<int> R;
            R.reserve(scratch_touched_targets.size());
            for (int v : scratch_touched_targets) {
                const uniqueDistT &best = cand_dist_val[v];
                if (best < getDist(v)) {
                    pred[v] = get<3>(best);
                    d[v] = get<0>(best);
                    path_sz[v] = get<1>(best);
                    root[v] = cand_owner_val[v];
                    if (best < B) {
                        R.push_back(v);
                        if (pivot_vis[v] != counter_pivot) {
                            pivot_vis[v] = counter_pivot;
                            vis.push_back(v);
                        }
                    }
                }
            }
            return R;
        }

        std::vector<int> R;
        R.reserve(Q.size() * 2 + 4);
        auto process_owner_list = [&](const std::vector<int> &owners) {
            for (int owner_u : owners) {
                const int st = owner_bucket_start[owner_u];
                const int en = st + owner_bucket_count[owner_u];
                for (int idx = st; idx < en; ++idx) {
                    const int u = scratch_bucket_vertices[idx];
                    const int cur_owner = root[u];
                    for (const auto &[v, w] : adj[u]) {
                        if (getDist(u, v, w) <= getDist(v)) {
                            updateDist(u, v, w);
                            if (getDist(v) < B) {
                                root[v] = cur_owner;
                                if (round_seen_stamp[v] != round_seen_token) {
                                    round_seen_stamp[v] = round_seen_token;
                                    R.push_back(v);
                                }
                                if (pivot_vis[v] != counter_pivot) {
                                    pivot_vis[v] = counter_pivot;
                                    vis.push_back(v);
                                }
                            }
                        }
                    }
                }
            }
        };
        process_owner_list(scratch_owner_nonpivot);
        process_owner_list(scratch_owner_pivot);
        return R;
    }

    std::pair<std::vector<int>, std::vector<int>> findPivots(uniqueDistT B, const std::vector<int> &S) { // Algorithm 1
        counter_pivot++;

        std::vector<int> vis;
        vis.reserve(2 * std::max<size_t>(1, static_cast<size_t>(k) * S.size()));
        for (int x : S) {
            vis.push_back(x);
            pivot_vis[x] = counter_pivot;
            root[x] = x;
            treesz[x] = 0;
        }

        std::vector<int> active = S;
        const size_t overflow_limit = static_cast<size_t>(k) * S.size();

        if (pivot_search_backend == pivot_search_backend_t::disabled || bf_prefix_steps >= k) {
            for (int i = 1; i <= k; ++i) {
                active = relaxRoundPrefixActual(active, B, vis);
                if (vis.size() > overflow_limit) {
                    return {S, vis};
                }
            }
            std::vector<int> P;
            P.reserve(vis.size() / std::max(1, k));
            for (int u : vis) treesz[root[u]]++;
            for (int u : S) if (treesz[u] >= k) P.push_back(u);
            return {P, vis};
        }

        const int prefix_rounds = std::max(0, std::min(bf_prefix_steps, k));
        for (int i = 1; i <= prefix_rounds; ++i) {
            active = relaxRoundPrefixActual(active, B, vis);
            if (vis.size() > overflow_limit) {
                return {S, vis};
            }
        }

        std::vector<int> pred_P;
        pred_P.reserve(S.size());
        std::vector<double> pred_prob(S.size(), 0.0);

        nextOwnerCountEpoch();
        int top_owner_count = 0;
        for (int u : vis) {
            const int owner_u = root[u];
            if (owner_count_stamp[owner_u] != owner_count_token) {
                owner_count_stamp[owner_u] = owner_count_token;
                owner_count_val[owner_u] = 0;
            }
            ++owner_count_val[owner_u];
            if (owner_count_val[owner_u] > top_owner_count) top_owner_count = owner_count_val[owner_u];
        }
        const double top_owner_mass = vis.empty() ? 0.0 : (static_cast<double>(top_owner_count) / static_cast<double>(vis.size()));

        nextStamp(pivot_member_token, pivot_member_stamp);
        if (use_countmin_predictor) {
            for (int idx = 0; idx < static_cast<int>(S.size()); ++idx) {
                const int s = S[idx];
                const int prefix_owner_count = (owner_count_stamp[s] == owner_count_token) ? owner_count_val[s] : 0;
                const auto decision = cm_model.infer(
                    cm_state,
                    static_cast<int>(l),
                    static_cast<int>(S.size()),
                    k,
                    prefix_owner_count,
                    idx,
                    static_cast<int>(vis.size()),
                    top_owner_mass
                );
                pred_prob[idx] = decision.prob;
                if (decision.pred) {
                    pred_P.push_back(s);
                    pivot_member_stamp[s] = pivot_member_token;
                }
            }
        }

        auto run_scheduled_round = [&](const std::vector<int> &Q) {
            switch (pivot_search_backend) {
                case pivot_search_backend_t::dedup:
                    return scheduleRoundDedupActual(Q, B, vis);
                case pivot_search_backend_t::npf:
                    return scheduleRoundNPFActual(Q, B, vis);
                case pivot_search_backend_t::ob:
                    return scheduleRoundOBActual(Q, B, vis, false);
                case pivot_search_backend_t::hybrid:
                    return scheduleRoundOBActual(Q, B, vis, true);
                case pivot_search_backend_t::disabled:
                default:
                    return relaxRoundPrefixActual(Q, B, vis);
            }
        };

        for (int i = prefix_rounds + 1; i <= k; ++i) {
            active = run_scheduled_round(active);
            if (vis.size() > overflow_limit) {
                return {S, vis};
            }
        }

        nextOwnerCountEpoch();
        for (int u : vis) {
            const int owner_u = root[u];
            if (owner_count_stamp[owner_u] != owner_count_token) {
                owner_count_stamp[owner_u] = owner_count_token;
                owner_count_val[owner_u] = 0;
            }
            ++owner_count_val[owner_u];
        }

        std::vector<int> P_final;
        P_final.reserve(std::max<size_t>(4, pred_P.size()));
        for (int s : S) {
            if (owner_count_stamp[s] == owner_count_token && owner_count_val[s] >= k) {
                P_final.push_back(s);
            }
        }

        if (use_countmin_predictor) {
            for (int idx = 0; idx < static_cast<int>(S.size()); ++idx) {
                const int s = S[idx];
                const int label = (owner_count_stamp[s] == owner_count_token && owner_count_val[s] >= k) ? 1 : 0;
                cm_model.update(cm_state, pred_prob[idx], label);
            }
        }

        return {P_final, vis};
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
 
    std::vector<std::unique_ptr<spp_lapq::batchPQ_iface<uniqueDistT>>> Ds;
    std::vector<short int> last_complete_lvl;
    std::pair<uniqueDistT, std::vector<int>> bmsspRec(short int l, uniqueDistT B, const std::vector<int> &S) { // Algorithm 3
        if(l == 0) {
            return baseCase(B, S[0]);
        }

        std::vector<int> P;
        std::vector<int> bellman_vis;
        if(use_ps_predictor) {
            const auto ps_decision = ps_model.infer(
                ps_state,
                static_cast<int>(l),
                0,
                static_cast<int>(S.size()),
                k,
                0,
                0,
                0
            );
            if(ps_decision.pred) {
                P = S;
                bellman_vis = S;
            } else {
                auto pivots = findPivots(B, S);
                P = std::move(pivots.first);
                bellman_vis = std::move(pivots.second);

                const int ps_label = (P.size() == S.size()) ? 1 : 0;
                ps_model.update(ps_state, ps_decision.prob, ps_label);
            }
        } else {
            auto pivots = findPivots(B, S);
            P = std::move(pivots.first);
            bellman_vis = std::move(pivots.second);
        }
 
        const long long batch_size = (1ll << ((l - 1) * t));
        auto &D = *Ds[l - 1];
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