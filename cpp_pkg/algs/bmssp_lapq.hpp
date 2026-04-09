#ifndef BMSSP_LAPQ_HPP
#define BMSSP_LAPQ_HPP

#include "bmssp.hpp"
#include "utils.hpp"

#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace spp_lapq {





template <typename uniqueDistT>
class batchPQ_iface {
public:
    double snip_split = 0.0;
    double snip_lower_bound = 0.0;
    double snip_block_insertion = 0.0;
    double snip_membership_check = 0.0;
    double snip_deletion = 0.0;

    virtual ~batchPQ_iface() = default;
    virtual void initialize(int M, uniqueDistT B) = 0;
    virtual int size() const = 0;
    virtual void insert(uniqueDistT x) = 0;
    virtual void batchPrepend(const std::vector<uniqueDistT>& v) = 0;
    virtual std::pair<uniqueDistT, std::vector<int>> pull() = 0;
    virtual void erase(int key) = 0;
};

template <typename uniqueDistT>
class bpq_batchPQ final : public batchPQ_iface<uniqueDistT> {
    template <typename V>
    using hash_map = spp::worst_case_hash_map<V>;

    using elementT = std::pair<int, uniqueDistT>;
    using block_t = std::list<elementT>;
    using block_it_t = typename std::list<block_t>::iterator;
    using elem_it_t = typename std::list<elementT>::iterator;

    struct CompareUB {
        bool operator()(const std::pair<uniqueDistT, block_it_t>& a,
                        const std::pair<uniqueDistT, block_it_t>& b) const {
            if (a.first != b.first) return a.first < b.first;
            return std::addressof(*a.second) < std::addressof(*b.second);
        }
    };

    block_it_t it_min_{};
    std::list<block_t> D0_;
    std::list<block_t> D1_;
    std::set<std::pair<uniqueDistT, block_it_t>, CompareUB> UBs_;

    int M_ = 0;
    int size_ = 0;
    uniqueDistT B_{};

    hash_map<uniqueDistT> actual_value_;
    hash_map<std::pair<block_it_t, elem_it_t>> where_is0_;
    hash_map<std::pair<block_it_t, elem_it_t>> where_is1_;

public:
    explicit bpq_batchPQ(int n)
        : actual_value_(n), where_is0_(n), where_is1_(n) {}

    void initialize(int M, uniqueDistT B) override {
        M_ = M;
        B_ = B;
        D0_.clear();
        D1_.clear();
        D1_.push_back(block_t{});
        UBs_.clear();
        UBs_.insert({B_, D1_.begin()});
        size_ = 0;

        actual_value_.clear();
        where_is0_.clear();
        where_is1_.clear();

        this->snip_split = 0.0;
        this->snip_lower_bound = 0.0;
        this->snip_block_insertion = 0.0;
        this->snip_membership_check = 0.0;
        this->snip_deletion = 0.0;
    }

    int size() const override { return size_; }

    void insert(uniqueDistT x) override {
        uniqueDistT b = x;
        int a = std::get<2>(b);

        auto it_exist = actual_value_.find(a);
        const bool exists = (it_exist != actual_value_.end());
        if (exists && it_exist->second > b) {
            delete_key(a);
        } else if (exists) {
            return;
        }

        auto it_UB_block = UBs_.lower_bound({b, it_min_});
        auto [ub, it_block] = *it_UB_block;
        (void)ub;

        auto it = it_block->insert(it_block->end(), {a, b});
        where_is1_[a] = {it_block, it};
        actual_value_[a] = b;
        ++size_;

        if (static_cast<int>(it_block->size()) > M_) {
            split(it_block);
        }
    }

    void batchPrepend(const std::vector<uniqueDistT>& v) override {
        std::list<elementT> l;
        for (const auto& x : v) {
            l.push_back({std::get<2>(x), x});
        }
        batchPrepend_list(l);
    }

    std::pair<uniqueDistT, std::vector<int>> pull() override {
        std::vector<elementT> s0, s1;
        s0.reserve(static_cast<std::size_t>(2 * std::max(1, M_)));
        s1.reserve(static_cast<std::size_t>(std::max(1, M_)));

        auto it_block = D0_.begin();
        while (it_block != D0_.end() && static_cast<int>(s0.size()) <= M_) {
            for (const auto& x : *it_block) s0.push_back(x);
            ++it_block;
        }

        it_block = D1_.begin();
        while (it_block != D1_.end() && static_cast<int>(s1.size()) <= M_) {
            for (const auto& x : *it_block) s1.push_back(x);
            ++it_block;
        }

        if (static_cast<int>(s0.size() + s1.size()) <= M_) {
            std::vector<int> ret;
            ret.reserve(s0.size() + s1.size());
            for (const auto& [a, b] : s0) {
                ret.push_back(a);
                delete_key(a);
            }
            for (const auto& [a, b] : s1) {
                ret.push_back(a);
                delete_key(a);
            }
            return {B_, std::move(ret)};
        }

        s0.insert(s0.end(), s1.begin(), s1.end());
        uniqueDistT med = selectKth(s0, M_);
        std::vector<int> ret;
        ret.reserve(static_cast<std::size_t>(M_));
        for (const auto& [a, b] : s0) {
            if (b < med) {
                ret.push_back(a);
                delete_key(a);
            }
        }
        return {med, std::move(ret)};
    }

    void erase(int key) override {
        if (actual_value_.find(key) != actual_value_.end()) {
            delete_key(key);
        }
    }

private:
    void delete_key(int a) {
        uniqueDistT b = actual_value_[a];

        auto it_w = where_is1_.find(a);
        if (it_w != where_is1_.end()) {
            auto [it_block, it] = it_w->second;
            it_block->erase(it);
            where_is1_.erase(a);

            if (it_block->empty()) {
                auto it_UB_block = UBs_.lower_bound({b, it_block});
                if (it_UB_block != UBs_.end() && !(it_UB_block->first == B_)) {
                    UBs_.erase(it_UB_block);
                    D1_.erase(it_block);
                }
            }
        } else {
            auto it0 = where_is0_.find(a);
            if (it0 != where_is0_.end()) {
                auto [it_block, it] = it0->second;
                it_block->erase(it);
                where_is0_.erase(a);
                if (it_block->empty()) {
                    D0_.erase(it_block);
                }
            }
        }

        actual_value_.erase(a);
        --size_;
    }

    static uniqueDistT selectKth(std::vector<elementT>& v, int k) {
        const auto comparator = [](const auto& a, const auto& b) {
            return a.second < b.second;
        };
        miniselect::median_of_ninthers_select(v.begin(), v.begin() + k, v.end(), comparator);
        return v[static_cast<std::size_t>(k)].second;
    }

    void split(block_it_t it_block) {
        const int sz = static_cast<int>(it_block->size());
        std::vector<elementT> v(it_block->begin(), it_block->end());
        uniqueDistT med = selectKth(v, sz / 2);

        auto pos = it_block;
        ++pos;
        auto new_block = D1_.insert(pos, block_t{});

        for (auto it = it_block->begin(); it != it_block->end();) {
            if (!(it->second < med)) {
                new_block->push_back(std::move(*it));
                auto it_new = std::prev(new_block->end());
                where_is1_[it_new->first] = {new_block, it_new};
                it = it_block->erase(it);
            } else {
                ++it;
            }
        }

        uniqueDistT UB1 = {std::get<0>(med), std::get<1>(med), std::get<2>(med), std::get<3>(med) - 1};
        auto it_lb = UBs_.lower_bound({UB1, it_min_});
        auto [UB2, aux] = *it_lb;
        (void)aux;

        UBs_.insert({UB1, it_block});
        UBs_.insert({UB2, new_block});
        UBs_.erase(it_lb);
    }

    void batchPrepend_list(const std::list<elementT>& l) {
        const int sz = static_cast<int>(l.size());
        if (sz == 0) return;

        if (sz <= M_) {
            D0_.push_front(block_t{});
            auto new_block = D0_.begin();

            for (const auto& x : l) {
                auto it = actual_value_.find(x.first);
                const bool exists = (it != actual_value_.end());
                if (exists && it->second > x.second) {
                    delete_key(x.first);
                } else if (exists) {
                    continue;
                }

                new_block->push_back(x);
                auto it_new = std::prev(new_block->end());
                where_is0_[x.first] = {new_block, it_new};
                actual_value_[x.first] = x.second;
                ++size_;
            }
            if (new_block->empty()) D0_.erase(new_block);
            return;
        }

        std::vector<elementT> v(l.begin(), l.end());
        uniqueDistT med = selectKth(v, sz / 2);

        std::list<elementT> less, great;
        for (const auto& [a, b] : l) {
            if (b < med) {
                less.push_back({a, b});
            } else if (med < b) {
                great.push_back({a, b});
            }
        }
        great.push_back({std::get<2>(med), med});

        batchPrepend_list(great);
        batchPrepend_list(less);
    }
};


template <typename uniqueDistT>
class lapq_batchPQ final : public batchPQ_iface<uniqueDistT> {
    using elementT = std::pair<int, uniqueDistT>;

    struct Block;
    struct IndexNode {
        Block* block = nullptr;
        uniqueDistT key{};
        std::vector<IndexNode*> next;
        std::vector<IndexNode*> prev;

        IndexNode() = default;
        IndexNode(Block* b, const uniqueDistT& k, int h)
            : block(b), key(k), next(static_cast<std::size_t>(h), nullptr), prev(static_cast<std::size_t>(h), nullptr) {}
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
    explicit lapq_batchPQ(int n)
        : actual_value_(), where0_(), where1_(), rng_(0x9E3779B97F4A7C15ULL ^ static_cast<std::uint64_t>(n)), head_() {
        actual_value_.reserve(static_cast<std::size_t>(n) * 2 + 1);
        where0_.reserve(static_cast<std::size_t>(n) * 2 + 1);
        where1_.reserve(static_cast<std::size_t>(n) * 2 + 1);
        head_.next.assign(MAX_LEVEL, nullptr);
        head_.prev.assign(MAX_LEVEL, nullptr);
    }

    lapq_batchPQ(const lapq_batchPQ&) = delete;
    lapq_batchPQ& operator=(const lapq_batchPQ&) = delete;
    lapq_batchPQ(lapq_batchPQ&&) = delete;
    lapq_batchPQ& operator=(lapq_batchPQ&&) = delete;

    ~lapq_batchPQ() override { clear_all(); }

    void initialize(int M, uniqueDistT B) override {
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

        this->snip_split = 0.0;
        this->snip_lower_bound = 0.0;
        this->snip_block_insertion = 0.0;
        this->snip_membership_check = 0.0;
        this->snip_deletion = 0.0;
    }

    int size() const override { return size_; }

    void insert(uniqueDistT x) override {
        const uniqueDistT b = x;
        const int a = std::get<2>(b);

        auto it_exist = actual_value_.find(a);
        const bool exists = (it_exist != actual_value_.end());

        Block* hint = nullptr;
        if (exists) {
            auto it_loc = where1_.find(a);
            if (it_loc != where1_.end() && it_loc->second.block != nullptr && it_loc->second.block->active) {
                hint = it_loc->second.block;
            }
        }
        if (hint == nullptr) {
            const int parent = std::get<3>(b);
            auto it_parent = where1_.find(parent);
            if (it_parent != where1_.end() && it_parent->second.block != nullptr && it_parent->second.block->active) {
                hint = it_parent->second.block;
            }
        }
        if (hint == nullptr || !hint->active) {
            hint = (last_insert_block_ != nullptr && last_insert_block_->active) ? last_insert_block_ : first_d1_block_;
        }

        if (exists && it_exist->second > b) {
            delete_key(a);
        } else if (exists) {
            return;
        }

        Block* block = lower_bound_block(b, hint);
        block->elems.emplace_back(a, b);
        auto elem_it = std::prev(block->elems.end());
        where1_[a] = {block, elem_it};
        actual_value_[a] = b;
        ++size_;
        last_search_block_ = block;
        last_insert_block_ = block;

        if (static_cast<int>(block->elems.size()) > M_) {
            split_block(block);
        }
    }

    void batchPrepend(const std::vector<uniqueDistT>& v) override {
        std::list<elementT> l;
        for (const auto& x : v) {
            l.emplace_back(std::get<2>(x), x);
        }
        batchPrepend_list(l);
    }

    std::pair<uniqueDistT, std::vector<int>> pull() override {
        std::vector<elementT> s0;
        std::vector<elementT> s1;
        s0.reserve(static_cast<std::size_t>(2 * std::max(1, M_)));
        s1.reserve(static_cast<std::size_t>(std::max(1, M_)));

        for (auto it = D0_.begin(); it != D0_.end() && static_cast<int>(s0.size()) <= M_; ++it) {
            for (const auto& x : (*it)->elems) {
                s0.push_back(x);
            }
        }
        for (auto it = D1_.begin(); it != D1_.end() && static_cast<int>(s1.size()) <= M_; ++it) {
            for (const auto& x : (*it)->elems) {
                s1.push_back(x);
            }
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
        ret.reserve(static_cast<std::size_t>(M_));
        for (const auto& [a, b] : s0) {
            if (b < med) {
                ret.push_back(a);
                delete_key(a);
            }
        }
        return {med, std::move(ret)};
    }

    void erase(int key) override {
        if (actual_value_.find(key) != actual_value_.end()) {
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
        max_level_ = 1;
        head_.next.assign(MAX_LEVEL, nullptr);
        head_.prev.assign(MAX_LEVEL, nullptr);
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
        if (hint == nullptr) {
            return predecessor_from_head(key);
        }

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
        return cur->prev.empty() || cur->prev[0] == nullptr ? const_cast<IndexNode*>(&head_) : cur->prev[0];
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

        std::vector<IndexNode*> update(static_cast<std::size_t>(MAX_LEVEL), &head_);
        IndexNode* cur = pred0 == nullptr ? &head_ : pred0;
        update[0] = cur;
        for (int lvl = 1; lvl < h; ++lvl) {
            while (cur != &head_ && cur->height() <= lvl) {
                cur = (cur->prev.empty() || cur->prev[0] == nullptr) ? &head_ : cur->prev[0];
            }
            update[static_cast<std::size_t>(lvl)] = cur == nullptr ? &head_ : cur;
        }

        auto* node = new IndexNode(block, key, h);
        block->idx = node;
        block->ub = key;
        for (int lvl = 0; lvl < h; ++lvl) {
            IndexNode* prev = update[static_cast<std::size_t>(lvl)];
            IndexNode* next = prev->next[static_cast<std::size_t>(lvl)];
            node->prev[static_cast<std::size_t>(lvl)] = prev;
            node->next[static_cast<std::size_t>(lvl)] = next;
            prev->next[static_cast<std::size_t>(lvl)] = node;
            if (next != nullptr) {
                next->prev[static_cast<std::size_t>(lvl)] = node;
            }
        }
    }

    void erase_index_node(Block* block) {
        IndexNode* node = block->idx;
        if (node == nullptr) {
            return;
        }
        for (int lvl = 0; lvl < node->height(); ++lvl) {
            IndexNode* prev = node->prev[static_cast<std::size_t>(lvl)];
            IndexNode* next = node->next[static_cast<std::size_t>(lvl)];
            if (prev != nullptr) {
                prev->next[static_cast<std::size_t>(lvl)] = next;
            }
            if (next != nullptr) {
                next->prev[static_cast<std::size_t>(lvl)] = prev;
            }
        }
        if (last_search_block_ == block) {
            last_search_block_ = (node->next[0] != nullptr) ? node->next[0]->block : nullptr;
        }
        if (last_insert_block_ == block) {
            last_insert_block_ = (node->prev[0] != nullptr && node->prev[0] != &head_) ? node->prev[0]->block : first_d1_block_;
        }
        delete node;
        block->idx = nullptr;
        while (max_level_ > 1 && head_.next[static_cast<std::size_t>(max_level_ - 1)] == nullptr) {
            --max_level_;
        }
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
        for (auto it = seq.begin(); it != seq.end(); ++it) {
            if (*it == ptr) {
                return it;
            }
        }
        return seq.end();
    }

    void delete_key(int a) {
        auto it_val = actual_value_.find(a);
        if (it_val == actual_value_.end()) {
            return;
        }

        auto it1 = where1_.find(a);
        if (it1 != where1_.end()) {
            Block* block = it1->second.block;
            block->elems.erase(it1->second.elem_it);
            where1_.erase(it1);

            if (block->elems.empty() && !(block->ub == B_)) {
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
            }
        } else {
            auto it0 = where0_.find(a);
            if (it0 != where0_.end()) {
                Block* block = it0->second.block;
                block->elems.erase(it0->second.elem_it);
                where0_.erase(it0);
                if (block->elems.empty()) {
                    auto it_block = find_block(D0_, block);
                    if (it_block != D0_.end()) {
                        D0_.erase(it_block);
                    }
                    block->active = false;
                }
            }
        }

        actual_value_.erase(it_val);
        --size_;
    }

    void split_block(Block* block) {
        const int sz = static_cast<int>(block->elems.size());
        std::vector<elementT> snapshot;
        snapshot.reserve(static_cast<std::size_t>(sz));
        for (const auto& e : block->elems) {
            snapshot.push_back(e);
        }
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
        if (block->idx != nullptr) {
            block->idx->key = block->ub;
        }

        insert_index_node(new_block, old_ub, block->idx);
        last_search_block_ = new_block;
        last_insert_block_ = new_block;
    }

    void batchPrepend_list(const std::list<elementT>& l) {
        const int sz = static_cast<int>(l.size());
        if (sz == 0) {
            return;
        }

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

            if (best.empty()) {
                return;
            }

            Block* block = make_block(false);
            D0_.push_front(block);
            for (const auto& [a, b] : best) {
                auto it_live = actual_value_.find(a);
                if (it_live != actual_value_.end()) {
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
        for (const auto& x : l) {
            v.push_back(x);
        }
        const uniqueDistT med = kth_value(v, sz / 2);

        std::list<elementT> less;
        std::list<elementT> great;
        for (const auto& [a, b] : l) {
            if (b < med) {
                less.emplace_back(a, b);
            } else if (med < b) {
                great.emplace_back(a, b);
            }
        }
        great.emplace_back(std::get<2>(med), med);

        batchPrepend_list(great);
        batchPrepend_list(less);
    }
};

template <typename wT>
class bmssp {
    enum class backend_t { bpq, lapq };

    int n = 0;
    int k = 0;
    int t = 0;
    int l = 0;

    std::vector<std::vector<std::pair<int, wT>>> ori_adj;
    std::vector<std::vector<std::pair<int, wT>>> adj;
    std::vector<wT> d;
    std::vector<int> pred;
    std::vector<int> path_sz;

    std::vector<int> node_map;
    std::vector<int> node_rev_map;

    bool cd_transfomed = false;
    backend_t backend_ = backend_t::lapq;

public:
    Stats stats;
    const wT oo = std::numeric_limits<wT>::max() / 10;

    static backend_t parse_backend(const std::string& backend) {
        if (backend == "lapq") return backend_t::lapq;
        if (backend == "bpq") return backend_t::bpq;
        throw std::invalid_argument("bmssp backend must be \"bpq\" or \"lapq\"");
    }

    explicit bmssp(int n_, const std::string& backend = "lapq") : n(n_), backend_(parse_backend(backend)) {
        ori_adj.assign(static_cast<std::size_t>(n), {});
    }

    template <typename Adj>
    explicit bmssp(const Adj& adj_, const std::string& backend = "lapq") : backend_(parse_backend(backend)) {
        n = static_cast<int>(adj_.size());
        ori_adj = adj_;
    }

    void addEdge(int a, int b, wT w) {
        ori_adj[a].emplace_back(b, w);
    }

    void prepare_graph(bool exec_constant_degree_trasnformation = false) {
        cd_transfomed = exec_constant_degree_trasnformation;

        std::vector<std::pair<int, int>> tmp_edges(static_cast<std::size_t>(n), {-1, -1});
        for (int i = 0; i < n; i++) {
            std::vector<std::pair<int, wT>> nw_adj;
            nw_adj.reserve(ori_adj[i].size());
            for (auto [j, w] : ori_adj[i]) {
                if (tmp_edges[j].first != i) {
                    nw_adj.emplace_back(j, w);
                    tmp_edges[j] = {i, static_cast<int>(nw_adj.size()) - 1};
                } else {
                    int id = tmp_edges[j].second;
                    nw_adj[static_cast<std::size_t>(id)].second = std::min(nw_adj[static_cast<std::size_t>(id)].second, w);
                }
            }
            ori_adj[i] = std::move(nw_adj);
        }
        tmp_edges.clear();

        if (!exec_constant_degree_trasnformation) {
            adj = std::move(ori_adj);
            node_map.resize(static_cast<std::size_t>(n));
            node_rev_map.resize(static_cast<std::size_t>(n));
            for (int i = 0; i < n; i++) {
                node_map[static_cast<std::size_t>(i)] = i;
                node_rev_map[static_cast<std::size_t>(i)] = i;
            }
        } else {
            int cnt = 0;
            std::vector<std::map<int, int>> edge_id(static_cast<std::size_t>(n));
            for (int i = 0; i < n; i++) {
                for (auto [j, w] : ori_adj[i]) {
                    (void)w;
                    if (edge_id[static_cast<std::size_t>(i)].find(j) == edge_id[static_cast<std::size_t>(i)].end()) {
                        edge_id[static_cast<std::size_t>(i)][j] = cnt++;
                        edge_id[static_cast<std::size_t>(j)][i] = cnt++;
                    }
                }
            }

            cnt++;
            adj.assign(static_cast<std::size_t>(cnt), {});
            node_map.resize(static_cast<std::size_t>(cnt));
            node_rev_map.resize(static_cast<std::size_t>(cnt));

            for (int i = 0; i < n; i++) {
                for (auto cur = edge_id[static_cast<std::size_t>(i)].begin(); cur != edge_id[static_cast<std::size_t>(i)].end(); ++cur) {
                    auto nxt = std::next(cur);
                    if (nxt == edge_id[static_cast<std::size_t>(i)].end()) {
                        nxt = edge_id[static_cast<std::size_t>(i)].begin();
                    }
                    adj[static_cast<std::size_t>(cur->second)].emplace_back(nxt->second, wT());
                    node_rev_map[static_cast<std::size_t>(cur->second)] = i;
                }
            }
            for (int i = 0; i < n; i++) {
                for (auto [j, w] : ori_adj[i]) {
                    adj[static_cast<std::size_t>(edge_id[static_cast<std::size_t>(i)][j])].emplace_back(edge_id[static_cast<std::size_t>(j)][i], w);
                }
                if (!edge_id[static_cast<std::size_t>(i)].empty()) {
                    node_map[static_cast<std::size_t>(i)] = edge_id[static_cast<std::size_t>(i)].begin()->second;
                } else {
                    node_map[static_cast<std::size_t>(i)] = cnt - 1;
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
        Ds.clear();
        Ds.reserve(static_cast<std::size_t>(l));
        for (int i = 0; i < l; ++i) {
            if (backend_ == backend_t::lapq) {
                Ds.push_back(std::make_unique<lapq_batchPQ<uniqueDistT>>(static_cast<int>(adj.size())));
            } else {
                Ds.push_back(std::make_unique<bpq_batchPQ<uniqueDistT>>(static_cast<int>(adj.size())));
            }
        }
    }

    std::pair<std::vector<wT>, std::vector<int>> execute(int s) {
        std::fill(d.begin(), d.end(), oo);
        std::fill(last_complete_lvl.begin(), last_complete_lvl.end(), static_cast<short int>(-1));
        std::fill(pivot_vis.begin(), pivot_vis.end(), -1);
        for (std::size_t i = 0; i < pred.size(); i++) {
            pred[i] = static_cast<int>(i);
        }

        s = toAnyCustomNode(s);
        d[static_cast<std::size_t>(s)] = 0;
        path_sz[static_cast<std::size_t>(s)] = 0;

        const int local_l = static_cast<int>(std::ceil(std::log2(adj.size()) / t));
        const uniqueDistT inf_dist = {oo, 0, 0, 0};
        bmsspRec(static_cast<short int>(local_l), inf_dist, {s});

        if (!cd_transfomed) {
            return {d, pred};
        }

        std::vector<wT> ret_distance(static_cast<std::size_t>(n));
        std::vector<int> ret_pred(static_cast<std::size_t>(n));
        for (int i = 0; i < n; i++) {
            ret_distance[static_cast<std::size_t>(i)] = d[static_cast<std::size_t>(toAnyCustomNode(i))];
            ret_pred[static_cast<std::size_t>(i)] = customToReal(getPred(toAnyCustomNode(i)));
        }
        return {ret_distance, ret_pred};
    }

    std::vector<int> get_shortest_path(int real_u, const std::vector<int>& real_pred) {
        if (!cd_transfomed) {
            int u = real_u;
            if (d[static_cast<std::size_t>(u)] == oo) {
                return {};
            }

            int cur_path_sz = std::get<1>(getDist(u)) + 1;
            std::vector<int> path(static_cast<std::size_t>(cur_path_sz));
            for (int i = cur_path_sz - 1; i >= 0; i--) {
                path[static_cast<std::size_t>(i)] = u;
                u = pred[static_cast<std::size_t>(u)];
            }
            return path;
        }

        int u = real_u;
        if (d[static_cast<std::size_t>(toAnyCustomNode(u))] == oo) {
            return {};
        }

        int max_path_sz = std::get<1>(getDist(toAnyCustomNode(u))) + 1;
        std::vector<int> path;
        path.reserve(static_cast<std::size_t>(max_path_sz));

        int oldu;
        do {
            path.push_back(u);
            oldu = u;
            u = real_pred[static_cast<std::size_t>(u)];
        } while (u != oldu);

        std::reverse(path.begin(), path.end());
        return path;
    }

private:
    inline int toAnyCustomNode(int real_id) const {
        return node_map[static_cast<std::size_t>(real_id)];
    }

    inline int customToReal(int id) const {
        return node_rev_map[static_cast<std::size_t>(id)];
    }

    int getPred(int u) const {
        int real_u = customToReal(u);

        int dad = u;
        do {
            dad = pred[static_cast<std::size_t>(dad)];
        } while (customToReal(dad) == real_u && pred[static_cast<std::size_t>(dad)] != dad);

        return dad;
    }

    template <typename T>
    bool isUnique(const std::vector<T>& v) {
        auto v2 = v;
        std::sort(v2.begin(), v2.end());
        v2.erase(std::unique(v2.begin(), v2.end()), v2.end());
        return v2.size() == v.size();
    }

    struct uniqueDistT : std::tuple<wT, int, int, int> {
        static constexpr wT SCALE = 1e10;
        static constexpr wT SCALE_INV = static_cast<wT>(1.0) / SCALE;

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
        return {d[static_cast<std::size_t>(u)] + w, path_sz[static_cast<std::size_t>(u)] + 1, v, u};
    }

    inline uniqueDistT getDist(int u) const {
        return {d[static_cast<std::size_t>(u)], path_sz[static_cast<std::size_t>(u)], u, pred[static_cast<std::size_t>(u)]};
    }

    void updateDist(int u, int v, wT w) {
        pred[static_cast<std::size_t>(v)] = u;
        d[static_cast<std::size_t>(v)] = d[static_cast<std::size_t>(u)] + w;
        path_sz[static_cast<std::size_t>(v)] = path_sz[static_cast<std::size_t>(u)] + 1;
    }

    std::vector<int> root;
    std::vector<short int> treesz;

    int counter_pivot = 0;
    std::vector<int> pivot_vis;

    std::pair<std::vector<int>, std::vector<int>> findPivots(uniqueDistT B, const std::vector<int>& S) {
        counter_pivot++;

        std::vector<int> vis;
        vis.reserve(static_cast<std::size_t>(2 * k * static_cast<int>(S.size())));

        for (int x : S) {
            vis.push_back(x);
            pivot_vis[static_cast<std::size_t>(x)] = counter_pivot;
        }

        std::vector<int> active = S;
        for (int x : S) {
            root[static_cast<std::size_t>(x)] = x;
            treesz[static_cast<std::size_t>(x)] = 0;
        }
        for (int i = 1; i <= k; i++) {
            std::vector<int> nw_active;
            nw_active.reserve(active.size() * 4);
            for (int u : active) {
                for (auto [v, w] : adj[static_cast<std::size_t>(u)]) {
                    if (getDist(u, v, w) <= getDist(v)) {
                        updateDist(u, v, w);
                        if (getDist(v) < B) {
                            root[static_cast<std::size_t>(v)] = root[static_cast<std::size_t>(u)];
                            nw_active.push_back(v);
                        }
                    }
                }
            }
            for (int x : nw_active) {
                if (pivot_vis[static_cast<std::size_t>(x)] != counter_pivot) {
                    pivot_vis[static_cast<std::size_t>(x)] = counter_pivot;
                    vis.push_back(x);
                }
            }
            if (vis.size() > static_cast<std::size_t>(k * static_cast<int>(S.size()))) {
                return {S, vis};
            }
            active = std::move(nw_active);
        }

        std::vector<int> P;
        P.reserve(vis.size() / std::max(1, k));
        for (int u : vis) {
            treesz[static_cast<std::size_t>(root[static_cast<std::size_t>(u)])]++;
        }
        for (int u : S) {
            if (treesz[static_cast<std::size_t>(u)] >= k) {
                P.push_back(u);
            }
        }

        return {P, vis};
    }

    std::pair<uniqueDistT, std::vector<int>> baseCase(uniqueDistT B, int x) {
        std::vector<int> complete;
        complete.reserve(static_cast<std::size_t>(k + 1));

        std::priority_queue<uniqueDistT, std::vector<uniqueDistT>, std::greater<uniqueDistT>> heap;
        heap.push(getDist(x));
        while (!heap.empty() && static_cast<int>(complete.size()) < k + 1) {
            auto du = heap.top();
            int u = std::get<2>(du);
            heap.pop();

            if (du > getDist(u)) {
                continue;
            }

            complete.push_back(u);
            for (auto [v, w] : adj[static_cast<std::size_t>(u)]) {
                auto new_dist = getDist(u, v, w);
                auto old_dist = getDist(v);
                if (new_dist <= old_dist && new_dist < B) {
                    updateDist(u, v, w);
                    heap.push(new_dist);
                }
            }
        }
        if (static_cast<int>(complete.size()) <= k) {
            return {B, complete};
        }

        uniqueDistT nB = getDist(complete.back());
        complete.pop_back();
        return {nB, complete};
    }

    std::vector<std::unique_ptr<batchPQ_iface<uniqueDistT>>> Ds;
    std::vector<short int> last_complete_lvl;

    std::pair<uniqueDistT, std::vector<int>> bmsspRec(short int level, uniqueDistT B, const std::vector<int>& S) {
        if (level == 0) {
            return baseCase(B, S[0]);
        }

        auto [P, bellman_vis] = findPivots(B, S);

        const long long batch_size = (1LL << ((level - 1) * t));
        auto& D = *Ds[static_cast<std::size_t>(level - 1)];
        D.initialize(static_cast<int>(batch_size), B);

        for (int p : P) {
            D.insert(getDist(p));
        }

        uniqueDistT last_complete_B = B;
        for (int p : P) {
            last_complete_B = std::min(last_complete_B, getDist(p));
        }

        std::vector<int> complete;
        const long long quota = static_cast<long long>(k) * (1LL << (level * t));
        complete.reserve(static_cast<std::size_t>(quota + static_cast<long long>(bellman_vis.size())));
        while (static_cast<long long>(complete.size()) < quota && D.size()) {
            auto [trying_B, miniS] = D.pull();
            auto [complete_B, nw_complete] = bmsspRec(level - 1, trying_B, miniS);

            complete.insert(complete.end(), nw_complete.begin(), nw_complete.end());

            std::vector<uniqueDistT> can_prepend;
            can_prepend.reserve(nw_complete.size() * 5 + miniS.size());
            for (int u : nw_complete) {
                D.erase(u);
                last_complete_lvl[static_cast<std::size_t>(u)] = level;
                for (auto [v, w] : adj[static_cast<std::size_t>(u)]) {
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
                if (complete_B <= getDist(x)) {
                    can_prepend.emplace_back(getDist(x));
                }
            }
            D.batchPrepend(can_prepend);
            last_complete_B = complete_B;
        }

        uniqueDistT retB;
        if (D.size() == 0) {
            retB = B;
        } else {
            retB = last_complete_B;
        }

        for (int x : bellman_vis) {
            if (last_complete_lvl[static_cast<std::size_t>(x)] != level && getDist(x) < retB) {
                complete.push_back(x);
            }
        }

        stats.snip_lower_bound += D.snip_lower_bound;
        stats.snip_split += D.snip_split;
        stats.snip_block_insertion += D.snip_block_insertion;
        stats.snip_membership_check += D.snip_membership_check;
        stats.snip_deletion += D.snip_deletion;

        return {retB, complete};
    }
};

} // namespace spp_lapq

#endif
