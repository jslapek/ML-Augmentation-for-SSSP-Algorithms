#include <bits/stdc++.h>

using namespace std;

namespace {

constexpr double RADIUS = 1.0;
constexpr long long WEIGHT_SCALE = 100000LL;
constexpr double PI = 3.141592653589793238462643383279502884;

struct Point {
    double x;
    double y;
};

static double lost_cap_area(double d) {
    if (d >= RADIUS) return 0.0;
    if (d <= 0.0) return PI * RADIUS * RADIUS / 2.0;
    const double x = d / RADIUS;
    return RADIUS * RADIUS * (acos(x) - x * sqrt(max(0.0, 1.0 - x * x)));
}

static double visible_disk_area(double y, double width) {
    const double full = PI * RADIUS * RADIUS;
    return full - lost_cap_area(y) - lost_cap_area(width - y);
}

static double average_visible_disk_area(double width) {
    if (width <= 0.0) {
        throw invalid_argument("strip_width must be positive");
    }
    // Simpson's rule: cheap, smooth, and accurate enough for parameter calibration.
    const int STEPS = 4096; // must be even
    const double h = width / STEPS;
    double sum = visible_disk_area(0.0, width) + visible_disk_area(width, width);
    for (int i = 1; i < STEPS; ++i) {
        const double y = i * h;
        sum += (i & 1 ? 4.0 : 2.0) * visible_disk_area(y, width);
    }
    return (h / 3.0) * sum / width;
}

static long long scaled_weight(double dx, double dy) {
    const double dist = sqrt(dx * dx + dy * dy);
    return max(1LL, llround(dist * WEIGHT_SCALE));
}

struct StreamGenerator {
    long long n;
    long long target_outdegree;
    double width;
    uint64_t seed;
    double effective_area;
    double geometric_outdegree;
    double rho;
    double length;
    double raw_threshold;

    explicit StreamGenerator(long long n_, long long d_, double width_, uint64_t seed_)
        : n(n_), target_outdegree(d_), width(width_), seed(seed_) {
        if (n <= 0) throw invalid_argument("number_of_vertices must be positive");
        if (target_outdegree < 2) throw invalid_argument("average_outdegree must be at least 2");
        if (width <= 0.0) throw invalid_argument("strip_width must be positive");

        effective_area = average_visible_disk_area(width);
        // In this narrow-strip regime, the planted forward backbone overlaps with a non-trivial fraction
        // of nearest-neighbour geometric edges. Empirically, subtracting only 0.5 (rather than 1.0)
        // keeps the realised average out-degree close to the requested target over the tested widths.
        geometric_outdegree = max(0.0, static_cast<double>(target_outdegree) - 0.5 + (n > 0 ? 0.5 / n : 0.0));
        rho = (geometric_outdegree <= 0.0 ? 1e-12 : geometric_outdegree / effective_area);
        length = static_cast<double>(n) / (rho * width);
    }

    double compute_total_raw_x() const {
        mt19937_64 gen(seed);
        exponential_distribution<double> exp01(1.0);
        double total = 0.0;
        for (long long i = 0; i < n; ++i) total += exp01(gen);
        return total;
    }

    template <class Callback>
    void stream_points(double total_raw_x, Callback callback) const {
        mt19937_64 gen(seed);
        exponential_distribution<double> exp01(1.0);
        uniform_real_distribution<double> ydis(0.0, width);

        const double scale = length / total_raw_x;
        double prefix = 0.0;
        for (long long i = 0; i < n; ++i) {
            prefix += exp01(gen);
            const Point p{prefix * scale, ydis(gen)};
            callback(i + 1, p); // 1-based labels to match the repo's DIMACS files.
        }
    }
};

} // namespace

signed main(int argc, char** argv) {
    if (argc < 5) {
        cout << "must have 4 arguments: number_of_vertices average_outdegree strip_width seed" << '\n';
        return 1;
    }

    const long long n = atoll(argv[1]);
    const long long average_outdegree = atoll(argv[2]);
    const double strip_width = atof(argv[3]);
    const uint64_t seed = static_cast<uint64_t>(atoll(argv[4]));

    StreamGenerator g(n, average_outdegree, strip_width, seed);
    const double total_raw_x = g.compute_total_raw_x();
    g.raw_threshold = total_raw_x / g.length; // raw-space equivalent of distance 1 in x after scaling.

    // Pass 1: count arcs.
    long long m = 0;
    if (n >= 2) m += (n - 1); // forward backbone along x-order (source 1 reaches all vertices).

    {
        mt19937_64 gen(seed);
        exponential_distribution<double> exp01(1.0);
        uniform_real_distribution<double> ydis(0.0, strip_width);
        deque<pair<long long, Point>> active;

        const double scale = g.length / total_raw_x;
        double prefix = 0.0;
        for (long long i = 0; i < n; ++i) {
            prefix += exp01(gen);
            Point cur{prefix * scale, ydis(gen)};

            while (!active.empty() && cur.x - active.front().second.x > RADIUS) active.pop_front();

            for (const auto& [j, p] : active) {
                const double dx = cur.x - p.x;
                const double dy = fabs(cur.y - p.y);
                if (dy <= RADIUS && dx * dx + dy * dy <= RADIUS * RADIUS) {
                    m += (j == i ? 1 : 2);
                }
            }

            active.push_back({i + 1, cur});
        }
    }

    cout << "p sp " << n << " " << m << '\n';

    // Pass 2: emit arcs.
    mt19937_64 gen(seed);
    exponential_distribution<double> exp01(1.0);
    uniform_real_distribution<double> ydis(0.0, strip_width);
    deque<pair<long long, Point>> active;
    optional<pair<long long, Point>> prev;

    const double scale = g.length / total_raw_x;
    double prefix = 0.0;
    for (long long i = 0; i < n; ++i) {
        prefix += exp01(gen);
        Point cur{prefix * scale, ydis(gen)};
        const long long id = i + 1;

        if (prev.has_value()) {
            const auto& [pid, pp] = *prev;
            const long long w = scaled_weight(cur.x - pp.x, cur.y - pp.y);
            cout << "a " << pid << ' ' << id << ' ' << w << '\n';
        }

        while (!active.empty() && cur.x - active.front().second.x > RADIUS) active.pop_front();

        for (const auto& [j, p] : active) {
            const double dx = cur.x - p.x;
            const double dy = fabs(cur.y - p.y);
            if (dy <= RADIUS && dx * dx + dy * dy <= RADIUS * RADIUS) {
                const long long w = scaled_weight(dx, dy);
                if (j == i) {
                    cout << "a " << id << ' ' << j << ' ' << w << '\n';
                } else {
                    cout << "a " << j << ' ' << id << ' ' << w << '\n';
                    cout << "a " << id << ' ' << j << ' ' << w << '\n';
                }
            }
        }

        active.push_back({id, cur});
        prev = {id, cur};
    }

    return 0;
}
