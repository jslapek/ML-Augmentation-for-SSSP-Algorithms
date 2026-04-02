#include <bits/stdc++.h>
#include <random>
#define int long long
using namespace std;

// Tailored-Bounded Geodesic Corridor (TBGC) graph generator
// Same API / output style as random-graph-generator.cpp:
//   ./tbgc-generator number_of_vertices average_outdegree max_weight seed
// Output: DIMACS directed graph
//   p sp n m
//   a u v w

signed main(signed argc, char **argv) {
    if (argc < 5) {
        cout << "must have 4 arguments: number_of_vertices average_outdegree max_weight seed" << endl;
        return 1;
    }

    int n = atoll(argv[1]);
    int average_outdegree = atoll(argv[2]);
    int max_weight = atoll(argv[3]);
    int seed = atoll(argv[4]);

    if (n <= 0) {
        cout << "number_of_vertices must be positive" << endl;
        return 1;
    }
    if (average_outdegree < 1) {
        cout << "average_outdegree must be at least 1" << endl;
        return 1;
    }
    if (max_weight < 1) {
        cout << "max_weight must be at least 1" << endl;
        return 1;
    }

    mt19937_64 gen(seed);
    auto random_integer = [&](int l, int r) -> int {
        uniform_int_distribution<uint64_t> dis((uint64_t)l, (uint64_t)r);
        return (int)dis(gen);
    };

    // Target edge count matches the existing random generator.
    int target_m = n * average_outdegree;
    int max_possible = n * (n - 1);
    target_m = min(target_m, max_possible);

    // Build a small number of layers so that the total light-path radius stays well below
    // the heavy-edge range; this creates the tailored bounded family PBMSSP should like.
    int corridor_layers;
    if (n == 1) corridor_layers = 0;
    else {
        int c_bound = max_weight / 8;
        corridor_layers = max<int>(2, (int)floor(sqrt((long double)n)));
        if (c_bound >= 2) corridor_layers = min(corridor_layers, c_bound);
        corridor_layers = min<int>(corridor_layers, n - 1);
    }

    // Partition vertices 2..n into corridor_layers layers as evenly as possible.
    vector<vector<int>> layers(max<int>(1, corridor_layers + 1));
    layers[0].push_back(1); // source
    if (n > 1) {
        int rem = n - 1;
        int base = rem / corridor_layers;
        int extra = rem % corridor_layers;
        int cur = 2;
        for (int i = 1; i <= corridor_layers; ++i) {
            int sz = base + (i <= extra ? 1 : 0);
            layers[i].reserve(sz);
            for (int j = 0; j < sz; ++j) layers[i].push_back(cur++);
        }
    }

    // Light edges should be very cheap; heavy edges should sit well above any all-light path.
    int light_hi;
    if (corridor_layers == 0) light_hi = 1;
    else light_hi = max<int>(1, max_weight / (4 * corridor_layers + 4));

    int heavy_lo;
    if (max_weight >= 4) heavy_lo = max<int>(light_hi + 1, (3 * max_weight) / 4);
    else heavy_lo = max<int>(light_hi + 1, max_weight);
    heavy_lo = min<int>(heavy_lo, max_weight);

    // Keep the number of low-weight edges intentionally tiny.
    int light_outdegree = min<int>(2, average_outdegree);

    vector<vector<int>> adj(n + 1);
    vector<unordered_set<int>> out(n + 1);
    vector<tuple<int,int,int>> edges;
    edges.reserve((size_t)target_m);

    auto canAddEdge = [&](int u, int v) -> bool {
        if (u < 1 || u > n || v < 1 || v > n || u == v) return false;
        return out[u].find(v) == out[u].end();
    };

    auto add_edge = [&](int u, int v, int w) -> bool {
        if (!canAddEdge(u, v)) return false;
        out[u].insert(v);
        adj[u].push_back(v);
        edges.push_back({u, v, w});
        return true;
    };

    // Step 1: guarantee reachability from source through a sparse light corridor.
    // Every vertex in layer i gets at least one incoming light edge from layer i-1.
    for (int i = 1; i <= corridor_layers; ++i) {
        auto &prev = layers[i - 1];
        auto &cur = layers[i];
        int prev_sz = (int)prev.size();
        for (int idx = 0; idx < (int)cur.size(); ++idx) {
            int v = cur[idx];
            int u = prev[random_integer(0, prev_sz - 1)];
            int w = random_integer(1, light_hi);
            bool ok = add_edge(u, v, w);
            if (!ok) {
                // Extremely rare duplicate fallback: deterministic scan.
                for (int pu : prev) {
                    if (add_edge(pu, v, w)) { ok = true; break; }
                }
            }
            assert(ok);
        }
    }

    // Step 2: add a few extra light corridor edges to create alternate cheap routes,
    // but keep them sparse relative to the heavy distractor population.
    if (average_outdegree >= 2) {
        for (int i = 0; i < corridor_layers; ++i) {
            auto &cur = layers[i];
            auto &nxt = layers[i + 1];
            if (nxt.empty()) continue;
            for (int u : cur) {
                while ((int)adj[u].size() < light_outdegree && (int)edges.size() < target_m) {
                    int v = nxt[random_integer(0, (int)nxt.size() - 1)];
                    int w = random_integer(1, light_hi);
                    int tries = 0;
                    bool ok = false;
                    while (tries < 8 && !(ok = add_edge(u, v, w))) {
                        v = nxt[random_integer(0, (int)nxt.size() - 1)];
                        ++tries;
                    }
                    if (!ok) break;
                }
            }
        }
    }

    // Helper to pick a destination with bias toward non-local layers, creating many heavy
    // distractor edges that are irrelevant to the true shortest paths.
    vector<int> vertex_layer(n + 1, 0);
    for (int i = 0; i <= corridor_layers; ++i) for (int v : layers[i]) vertex_layer[v] = i;

    auto random_vertex = [&]() -> int {
        return random_integer(1, n);
    };

    auto pick_heavy_target = [&](int u) -> int {
        int lu = vertex_layer[u];
        // Prefer targets outside the immediate next layer to avoid creating accidental cheap structure.
        for (int tries = 0; tries < 24; ++tries) {
            int v = random_vertex();
            if (v == u) continue;
            int lv = vertex_layer[v];
            if (lv == lu + 1 && corridor_layers >= 1) continue;
            if (canAddEdge(u, v)) return v;
        }
        // Fallback: any non-neighbor distinct vertex.
        int start = random_integer(1, n);
        for (int t = 0; t < n; ++t) {
            int v = 1 + ((start - 1 + t) % n);
            if (canAddEdge(u, v)) return v;
        }
        return -1;
    };

    // Step 3: fill the remaining budget with heavy distractor edges.
    // These are deliberately much heavier than any all-light corridor path.
    while ((int)edges.size() < target_m) {
        int u = random_vertex();
        int v = pick_heavy_target(u);
        if (v == -1) break; // graph already saturated
        int w = random_integer(heavy_lo, max_weight);
        bool ok = add_edge(u, v, w);
        (void)ok;
    }

    cout << "p sp " << n << " " << edges.size() << '\n';
    for (auto [u, v, w] : edges) {
        cout << "a " << u << " " << v << " " << w << '\n';
    }

    // Validate reachability from source 1.
    vector<bool> vis(n + 1, false);
    queue<int> q;
    q.push(1);
    vis[1] = true;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (!vis[v]) {
                vis[v] = true;
                q.push(v);
            }
        }
    }
    assert(accumulate(vis.begin(), vis.end(), 0LL) == n);

    return 0;
}
