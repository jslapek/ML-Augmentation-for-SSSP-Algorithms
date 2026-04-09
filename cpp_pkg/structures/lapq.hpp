#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lapq {

template <typename uniqueDistT>
class batchPQ {
    using elementT = std::pair<int, uniqueDistT>;
    using clockT = std::chrono::steady_clock;

    struct Block;
    struct IndexNode {
        Block* block = nullptr;
        uniqueDistT key{};
        std::vector<IndexNode*> next;
        std::vector<IndexNode*> prev;

        IndexNode() = default;
        IndexNode(Block* b, const uniqueDistT& k, int h) : block(b), key(k), next(h, nullptr), prev(h, nullptr) {}
        int height() const { return static_cast<int>(next.size()); }
    };

    struct Block {
        std::list<elementT> elems;
        uniqueDistT ub{};
        bool in_d1 = false;
        bool active = true;
        IndexNode* idx = nullptr;
    };

    using block_seq_t = std::list<Block*>;
    using elem_it_t = typename std::list<elementT>::iterator;
    using block_it_t = typename block_seq_t::iterator;

    struct Location {
        Block* block = nullptr;
        elem_it_t elem_it{};
    };

    struct Timer {
        clockT::time_point t0;
        Timer() : t0(clockT::now()) {}
        void reset() { t0 = clockT::now(); }
        double elapsed_ms() const {
            return std::chrono::duration<double, std::milli>(clockT::now() - t0).count();
        }
    };

    static constexpr int MAX_LEVEL = 24;

    block_seq_t D0_;
    block_seq_t D1_;
    std::vector<std::unique_ptr<Block>> arena_;

    int M_ = 0;
    int size_ = 0;
    uniqueDistT B_{};

    std::unordered_map<int, uniqueDistT> actual_value_;
    std::unordered_map<int, Location> where0_;
    std::unordered_map<int, Location> where1_;

    std::mt19937_64 rng_;
    int max_level_ = 1;
    IndexNode head_;

    Block* first_d1_block_ = nullptr;
    Block* last_search_block_ = nullptr;
    Block* last_insert_block_ = nullptr;

public:
    double snip_split = 0.0;
    double snip_lower_bound = 0.0;
    double snip_block_insertion = 0.0;
    double snip_membership_check = 0.0;
    double snip_deletion = 0.0;
    bool time_delete = false;

    explicit batchPQ(int n)
        : actual_value_(), where0_(), where1_(), rng_(0x9E3779B97F4A7C15ULL ^ static_cast<std::uint64_t>(n)), head_() {
        actual_value_.reserve(static_cast<std::size_t>(n) * 2 + 1);
        where0_.reserve(static_cast<std::size_t>(n) * 2 + 1);
        where1_.reserve(static_cast<std::size_t>(n) * 2 + 1);
        head_.next.assign(MAX_LEVEL, nullptr);
        head_.prev.assign(MAX_LEVEL, nullptr);
    }

    void initialize(int M, uniqueDistT B) {
        clear_all();
        M_ = M;
        B_ = B;
        size_ = 0;
        max_level_ = 1;
        head_.next.assign(MAX_LEVEL, nullptr);
        head_.prev.assign(MAX_LEVEL, nullptr);

        Block* sentinel = make_block(true, B_);
        D1_.push_back(sentinel);
        first_d1_block_ = sentinel;
        insert_index_node(sentinel, B_, &head_);
        last_search_block_ = sentinel;
        last_insert_block_ = sentinel;

        snip_split = 0.0;
        snip_lower_bound = 0.0;
        snip_block_insertion = 0.0;
        snip_membership_check = 0.0;
        snip_deletion = 0.0;
        time_delete = false;
    }

    int size() { return size_; }

    void insert(uniqueDistT x) {
        const uniqueDistT b = x;
        const int a = std::get<2>(b);

        Timer timer;
        auto it_exist = actual_value_.find(a);
        const bool exists = (it_exist != actual_value_.end());
        snip_membership_check += timer.elapsed_ms();

        Block* hint = nullptr;
        if (exists) {
            auto it_loc = where1_.find(a);
            if (it_loc != where1_.end() && it_loc->second.block->active) {
                hint = it_loc->second.block;
            }
        }
        if (hint == nullptr) {
            const int parent = std::get<3>(b);
            auto it_parent = where1_.find(parent);
            if (it_parent != where1_.end() && it_parent->second.block->active) {
                hint = it_parent->second.block;
            }
        }
        if (hint == nullptr || !hint->active) {
            hint = (last_insert_block_ != nullptr && last_insert_block_->active) ? last_insert_block_ : first_d1_block_;
        }

        if (exists && it_exist->second > b) {
            time_delete = true;
            delete_key(a);
        } else if (exists) {
            return;
        }

        timer.reset();
        Block* block = lower_bound_block(b, hint);
        snip_lower_bound += timer.elapsed_ms();

        timer.reset();
        block->elems.emplace_back(a, b);
        auto elem_it = std::prev(block->elems.end());
        where1_[a] = {block, elem_it};
        actual_value_[a] = b;
        ++size_;
        last_search_block_ = block;
        last_insert_block_ = block;
        snip_block_insertion += timer.elapsed_ms();

        if (static_cast<int>(block->elems.size()) > M_) {
            split_block(block);
        }
    }

    void batchPrepend(const std::vector<uniqueDistT>& v) {
        std::list<elementT> l;
        for (const auto& x : v) l.emplace_back(std::get<2>(x), x);
        batchPrepend_list(l);
    }

    std::pair<uniqueDistT, std::vector<int>> pull() {
        time_delete = false;
        std::vector<elementT> s0;
        std::vector<elementT> s1;
        s0.reserve(static_cast<std::size_t>(2 * std::max(1, M_)));
        s1.reserve(static_cast<std::size_t>(std::max(1, M_)));

        for (auto it = D0_.begin(); it != D0_.end() && static_cast<int>(s0.size()) <= M_; ++it) {
            for (const auto& x : (*it)->elems) s0.push_back(x);
        }
        for (auto it = D1_.begin(); it != D1_.end() && static_cast<int>(s1.size()) <= M_; ++it) {
            for (const auto& x : (*it)->elems) s1.push_back(x);
        }

        if (static_cast<int>(s0.size() + s1.size()) <= M_) {
            std::vector<int> ret;
            ret.reserve(s0.size() + s1.size());
            for (const auto& [a, _] : s0) {
                ret.push_back(a);
                delete_key(a);
            }
            for (const auto& [a, _] : s1) {
                ret.push_back(a);
                delete_key(a);
            }
            return {B_, std::move(ret)};
        }

        s0.insert(s0.end(), s1.begin(), s1.end());
        const uniqueDistT med = kth_value(s0, M_);
        std::vector<int> ret;
        ret.reserve(M_);
        for (const auto& [a, b] : s0) {
            if (b < med) {
                ret.push_back(a);
                delete_key(a);
            }
        }
        return {med, std::move(ret)};
    }

    inline void erase(int key) {
        if (actual_value_.find(key) != actual_value_.end()) {
            time_delete = true;
            delete_key(key);
        }
    }

private:
    void clear_all() {
        D0_.clear();
        D1_.clear();
        for (auto& blk : arena_) {
            if (blk && blk->idx) {
                delete blk->idx;
                blk->idx = nullptr;
            }
        }
        arena_.clear();
        actual_value_.clear();
        where0_.clear();
        where1_.clear();
        first_d1_block_ = nullptr;
        last_search_block_ = nullptr;
        last_insert_block_ = nullptr;
    }

    Block* make_block(bool in_d1, const uniqueDistT& ub = uniqueDistT{}) {
        arena_.push_back(std::make_unique<Block>());
        Block* b = arena_.back().get();
        b->ub = ub;
        b->in_d1 = in_d1;
        b->active = true;
        b->idx = nullptr;
        return b;
    }

    int random_height() {
        std::uint64_t bits = rng_();
        int h = 1;
        while ((bits & 1ULL) && h < MAX_LEVEL) {
            ++h;
            bits >>= 1ULL;
        }
        return h;
    }

    IndexNode* predecessor_from_head(const uniqueDistT& key) const {
        IndexNode* cur = const_cast<IndexNode*>(&head_);
        for (int lvl = max_level_ - 1; lvl >= 0; --lvl) {
            while (cur->next[lvl] != nullptr && cur->next[lvl]->key < key) {
                cur = cur->next[lvl];
            }
        }
        return cur;
    }

    IndexNode* predecessor_from_hint(IndexNode* hint, const uniqueDistT& key) const {
        if (hint == nullptr) return predecessor_from_head(key);

        if (hint->key < key) {
            IndexNode* cur = hint;
            for (int lvl = cur->height() - 1; lvl >= 0; --lvl) {
                while (lvl < cur->height() && cur->next[lvl] != nullptr && cur->next[lvl]->key < key) {
                    cur = cur->next[lvl];
                }
            }
            return cur;
        }

        IndexNode* cur = hint;
        for (int lvl = cur->height() - 1; lvl >= 0; --lvl) {
            while (lvl < cur->height() && cur->prev[lvl] != nullptr && cur->prev[lvl] != &head_ && !(cur->prev[lvl]->key < key)) {
                cur = cur->prev[lvl];
            }
        }
        return cur->prev[0] == nullptr ? const_cast<IndexNode*>(&head_) : cur->prev[0];
    }

    Block* lower_bound_block(const uniqueDistT& key, Block* hint_block) {
        IndexNode* pred = nullptr;
        if (hint_block != nullptr && hint_block->active && hint_block->idx != nullptr) {
            pred = predecessor_from_hint(hint_block->idx, key);
        } else if (last_search_block_ != nullptr && last_search_block_->active && last_search_block_->idx != nullptr) {
            pred = predecessor_from_hint(last_search_block_->idx, key);
        } else {
            pred = predecessor_from_head(key);
        }

        IndexNode* succ = pred->next[0];
        return succ != nullptr ? succ->block : D1_.back();
    }

    void insert_index_node(Block* block, const uniqueDistT& key, IndexNode* pred0) {
        const int h = random_height();
        max_level_ = std::max(max_level_, h);

        std::vector<IndexNode*> update(MAX_LEVEL, &head_);
        IndexNode* cur = pred0 == nullptr ? &head_ : pred0;
        update[0] = cur;
        for (int lvl = 1; lvl < h; ++lvl) {
            while (cur != &head_ && cur->height() <= lvl) cur = cur->prev[0] == nullptr ? &head_ : cur->prev[0];
            update[lvl] = cur == nullptr ? &head_ : cur;
        }

        auto* node = new IndexNode(block, key, h);
        block->idx = node;
        block->ub = key;
        for (int lvl = 0; lvl < h; ++lvl) {
            IndexNode* prev = update[lvl];
            IndexNode* next = prev->next[lvl];
            node->prev[lvl] = prev;
            node->next[lvl] = next;
            prev->next[lvl] = node;
            if (next != nullptr) next->prev[lvl] = node;
        }
    }

    void erase_index_node(Block* block) {
        IndexNode* node = block->idx;
        if (node == nullptr) return;
        for (int lvl = 0; lvl < node->height(); ++lvl) {
            IndexNode* prev = node->prev[lvl];
            IndexNode* next = node->next[lvl];
            if (prev != nullptr) prev->next[lvl] = next;
            if (next != nullptr) next->prev[lvl] = prev;
        }
        if (last_search_block_ == block) {
            last_search_block_ = (node->next[0] != nullptr) ? node->next[0]->block : nullptr;
        }
        if (last_insert_block_ == block) {
            last_insert_block_ = (node->prev[0] != nullptr && node->prev[0] != &head_) ? node->prev[0]->block : first_d1_block_;
        }
        delete node;
        block->idx = nullptr;
        while (max_level_ > 1 && head_.next[max_level_ - 1] == nullptr) --max_level_;
    }

    static uniqueDistT kth_value(std::vector<elementT>& v, int k) {
        std::nth_element(v.begin(), v.begin() + k, v.end(), [](const elementT& a, const elementT& b) {
            return a.second < b.second;
        });
        return v[static_cast<std::size_t>(k)].second;
    }

    static uniqueDistT make_left_upper_bound(const uniqueDistT& med) {
        return uniqueDistT{std::get<0>(med), std::get<1>(med), std::get<2>(med), std::get<3>(med) - 1};
    }

    block_it_t find_block(block_seq_t& seq, Block* ptr) {
        for (auto it = seq.begin(); it != seq.end(); ++it) if (*it == ptr) return it;
        return seq.end();
    }

    void delete_key(int a) {
        Timer timer;
        auto it_val = actual_value_.find(a);
        if (it_val == actual_value_.end()) return;

        auto it1 = where1_.find(a);
        if (it1 != where1_.end()) {
            Block* block = it1->second.block;
            block->elems.erase(it1->second.elem_it);
            where1_.erase(it1);

            if (block->elems.empty() && !(block->ub == B_)) {
                if (time_delete) snip_deletion += timer.elapsed_ms();
                timer.reset();
                erase_index_node(block);
                auto it_block = find_block(D1_, block);
                if (it_block != D1_.end()) {
                    if (first_d1_block_ == block) {
                        auto next_it = std::next(it_block);
                        first_d1_block_ = (next_it == D1_.end() ? nullptr : *next_it);
                    }
                    D1_.erase(it_block);
                }
                block->active = false;
                if (time_delete) snip_lower_bound += timer.elapsed_ms();
                timer.reset();
            }
        } else {
            auto it0 = where0_.find(a);
            if (it0 != where0_.end()) {
                Block* block = it0->second.block;
                block->elems.erase(it0->second.elem_it);
                where0_.erase(it0);
                if (block->elems.empty()) {
                    auto it_block = find_block(D0_, block);
                    if (it_block != D0_.end()) D0_.erase(it_block);
                    block->active = false;
                }
            }
        }

        actual_value_.erase(it_val);
        --size_;
        if (time_delete) snip_deletion += timer.elapsed_ms();
    }

    void split_block(Block* block) {
        Timer timer;
        const int sz = static_cast<int>(block->elems.size());
        std::vector<elementT> snapshot;
        snapshot.reserve(static_cast<std::size_t>(sz));
        for (const auto& e : block->elems) snapshot.push_back(e);
        const uniqueDistT med = kth_value(snapshot, sz / 2);
        const uniqueDistT old_ub = block->ub;

        Block* new_block = make_block(true, old_ub);
        auto it_block = find_block(D1_, block);
        D1_.insert(std::next(it_block), new_block);

        for (auto it = block->elems.begin(); it != block->elems.end();) {
            if (!(it->second < med)) {
                new_block->elems.push_back(std::move(*it));
                auto new_elem_it = std::prev(new_block->elems.end());
                where1_[new_elem_it->first] = {new_block, new_elem_it};
                it = block->elems.erase(it);
            } else {
                ++it;
            }
        }

        block->ub = make_left_upper_bound(med);
        if (block->idx != nullptr) block->idx->key = block->ub;

        insert_index_node(new_block, old_ub, block->idx);
        last_search_block_ = new_block;
        last_insert_block_ = new_block;
        snip_split += timer.elapsed_ms();
    }

    void batchPrepend_list(const std::list<elementT>& l) {
        const int sz = static_cast<int>(l.size());
        if (sz == 0) return;

        if (sz <= M_) {
            std::unordered_map<int, uniqueDistT> best;
            best.reserve(static_cast<std::size_t>(sz) * 2 + 1);
            for (const auto& x : l) {
                auto it_live = actual_value_.find(x.first);
                if (it_live != actual_value_.end() && !(x.second < it_live->second)) {
                    continue;
                }
                auto it_best = best.find(x.first);
                if (it_best == best.end() || x.second < it_best->second) {
                    best[x.first] = x.second;
                }
            }

            if (best.empty()) return;

            Block* block = make_block(false);
            D0_.push_front(block);
            for (const auto& [a, b] : best) {
                auto it_live = actual_value_.find(a);
                if (it_live != actual_value_.end()) {
                    time_delete = false;
                    delete_key(a);
                }
                block->elems.emplace_back(a, b);
                auto elem_it = std::prev(block->elems.end());
                where0_[a] = {block, elem_it};
                actual_value_[a] = b;
                ++size_;
            }
            return;
        }

        std::vector<elementT> v;
        v.reserve(static_cast<std::size_t>(sz));
        for (const auto& x : l) v.push_back(x);
        const uniqueDistT med = kth_value(v, sz / 2);

        std::list<elementT> less;
        std::list<elementT> great;
        for (const auto& [a, b] : l) {
            if (b < med) less.emplace_back(a, b);
            else if (med < b) great.emplace_back(a, b);
        }
        great.emplace_back(std::get<2>(med), med);

        batchPrepend_list(great);
        batchPrepend_list(less);
    }
};

} // namespace spp_timed

