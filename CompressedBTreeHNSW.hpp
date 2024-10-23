#ifndef RANGEHNSW_RANGEHNSW_HPP
#define RANGEHNSW_RANGEHNSW_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
//#include <sys/resource.h>

//#include "hnswlib/hnswlib.h"
#include "HNSW.hpp"
#define BTREE_M 4
#define BTREE_D 2

using namespace hnswlib;
class RangeHNSW {
public:
    RangeHNSW(
            int d,
            size_t eleNum,
            float* vecData,
            int* keyList,
            int* valueList,
            int th,
            int m,
            float a,
            int ef_con
            ):
            threshold(th),M(m),ef_construction(ef_con), space(d), dim(d), alpha(a){

        M0 = M * 2;
        nM = M * alpha;
        nM0 = M0 * alpha;
        extendLevel = 1 - 1.0 / alpha;
        maxLevel = floor(log((float)eleNum / (float)threshold) / log(BTREE_D));


        data_size_ = space.get_data_size();
        fstdistfunc_ = space.get_dist_func();
        dist_func_param_ = space.get_dist_func_param();

        keyList_ = new int[eleNum];
        valueList_ = new int[eleNum];
        vecData_ = (char*)(new float[eleNum * dim]);
        memcpy(keyList_,keyList, eleNum * sizeof(int));
        memcpy(valueList_,valueList, eleNum * sizeof(int));
        memcpy(vecData_,vecData, eleNum * dim * sizeof(float));

        linkList_level0.resize(eleNum);
        linklist.resize(eleNum);
        internalLevel.resize(eleNum);
        belongTo.resize(eleNum);
        visitedTag.resize(eleNum);

        for(int i = 0; i < eleNum; i++){
            belongTo[i].resize(maxLevel + 1);
        }

        mult_ = 1 / log(1.0 * nM);
        revSize_ = 1.0 / mult_;

        sizeLinkList = (nM * sizeof(tableint) + sizeof(linklistsizeint));
        sizeLinkList0 = (nM0 * sizeof(tableint) + sizeof(linklistsizeint));

        root = buildTree(0,eleNum - 1, maxLevel);
        std::vector<int> tmpList;
        tmpList.resize(eleNum);
        for(int i = 0; i < eleNum; i++) tmpList[i] = i;

        std::random_device rd;  // 用于获取随机数种子
        std::mt19937 g(rd());    // 初始化随机数生成器

        shuffle(tmpList.begin(),tmpList.end(),g);

        space = hnswlib::L2Space(dim);
        for(int id = 0; id < eleNum; id++){
            int i = tmpList[id];
            key2Id[keyList[i]] = i;
            internalLevel[i] = getRandomLevel(mult_);
            if(internalLevel[i] > 0) {
                linklist[i] = (char *) malloc(
                        internalLevel[i] * (maxLevel + 1) * (nM * sizeof(tableint) + sizeof(linklistsizeint)));
            }
            linkList_level0[i] = (char *) malloc((maxLevel + 1) *(nM0 * sizeof (tableint) + sizeof(linklistsizeint)));
            addPoint(i,internalLevel[i]);
        }

    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> queryRange(float *vecData, int rangeL, int rangeR, int k,int ef_s){
        Filter *filter = new Filter(this, rangeL, rangeR);
        auto nodes = findNode(root,rangeL, rangeR);
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> top_candidates;

        if(nodes.second == nullptr) {
            tableint e1 = findEntry(vecData,nodes.first);
            top_candidates = searchBaseLayer0(e1,0, nodes.first->exLevel,0, vecData, ef_s, filter);
        }
        else if(nodes.first == nullptr) {
            tableint e1 = findEntry(vecData,nodes.second);
            top_candidates = searchBaseLayer0(e1,0, nodes.second->exLevel,0, vecData, ef_s, filter);
        }
        else {
            tableint e1 = findEntry(vecData, nodes.first);
            tableint e2 = findEntry(vecData, nodes.second);
            top_candidates = searchBaseLayer0(e1, e2, nodes.first->exLevel, nodes.second->entryPoint, vecData, ef_s, filter);
        }


        std::priority_queue<std::pair<float, labeltype >> result;
        while (top_candidates.size() > 0) {
            std::pair<float, tableint> rez = top_candidates.top();
            result.push(std::pair<float, labeltype>(rez.first, keyList_[rez.second]));
            top_candidates.pop();
        }
        return result;
    }

private:

    size_t sizeLinkList;
    size_t sizeLinkList0;
    int extendLevel = 0;
    struct node{
//        hnswlib::HNSW<float> *hnsw;
        int entryPoint = -1;
        int keynum = BTREE_D - 1;
        int key[BTREE_M-1];
        struct node* child[BTREE_M];
        int L, R;
        int exLevel; //level in tree
        int maxInternalLevel = 0; //level in hnsw

        node(){}
        node(int l,int r):L(l),R(r){}

    };

    struct CompareByFirst {
        constexpr bool operator()(std::pair<float, tableint> const& a,
                                  std::pair<float, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    class Filter: public hnswlib::BaseFilterFunctor {
    public:
        Filter(RangeHNSW* rHnsw, int rL, int rR):rangeHnsw(rHnsw),rangeL(rL), rangeR(rR){}

        bool operator()(hnswlib::labeltype id) override{
            return (rangeHnsw->valueList_[id] >= rangeL) && (rangeHnsw->valueList_[id] <= rangeR);
        }
    private:
        RangeHNSW *rangeHnsw;
        int rangeL,rangeR;
    };


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    hnswlib::L2Space space;
    size_t data_size_{0};

    DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_{nullptr};

    node* root;
    char* vecData_;
    int* keyList_;
    int* valueList_;
    int threshold;
    int M,M0;
    int nM,nM0;
    int ef_construction;
    int dim;
    std::default_random_engine level_generator_;


    float alpha;
    double mult_{0.0}, revSize_{0.0};

    int maxLevel;

    std::vector<char *> linkList_level0;
    std::vector<char *> linklist;
    std::vector<int> internalLevel;

    std::vector<std::pair<float, tableint> > visitedList;
    std::vector<short int > visitedTag;
    int tag = 0;

    std::unordered_map<int,int> key2Id;

    std::vector<std::vector<node *>> belongTo;



    node* buildTree(int L,int R, int exLevel){
        node* newNode = new node(valueList_[L],valueList_[R]);
        newNode->exLevel = exLevel;
        for(int i = L; i <= R; i++)
            belongTo[i][exLevel] = newNode;

        if(exLevel == 0)
            return newNode;
        int len = (R-L+1) / BTREE_D;
        for(int i = 0; i <BTREE_D - 1; i++){
            newNode->key[i] = L + len*(i+1) -1;
            newNode->child[i] = buildTree(L + len * i, L + len * (i+1) -1, exLevel - 1);
        }
        newNode->child[BTREE_D - 1] = buildTree(L + len * (BTREE_D - 1), R, exLevel - 1);
        return newNode;
    }

    std::pair<node*, node*> findNode(node* node,int rangeL, int rangeR){
        if(node->exLevel == 0){
            return {node, nullptr};
        }

        if((rangeL <= node->L && rangeR >= node->key[0])||
                (rangeL <= node->key[node->keynum - 1] && rangeR >= node->R))
            return {node, nullptr};
        for(int i = 0; i < node->keynum - 1; i++){
            if(rangeL <= node->key[i]) {
                if (rangeR >= node->key[i + 1])
                    return {node, nullptr};
                if (rangeR <= node->key[i]) {
                    return findNode(node->child[i], rangeL, rangeR);
                }
                return {findNode(node->child[i],rangeL, rangeR).first, findNode(node->child[i + 1],rangeL, rangeR).first};
            }
        }
        return findNode(node->child[node->keynum],rangeL, rangeR);
    }

    void getNeighborsByHeuristic2(
            std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, CompareByFirst> &top_candidates,
            const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<float, tableint>> queue_closest;
        std::vector<std::pair<float, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<float, tableint> curent_pair = queue_closest.top();
            float dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<float, tableint> second_pair : return_list) {
                float curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                     getDataByInternalId(curent_pair.second),
                                     dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<float, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    inline char *getDataByInternalId(tableint internal_id) const {
        return (char*)(vecData_ + internal_id * data_size_);
    }

    std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer, int exLevel) {
        tag ++;

        std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, CompareByFirst> candidateSet;

        float lowerBound;

        float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidateSet.emplace(-dist, ep_id);

        visitedTag[ep_id] = tag;

        while (!candidateSet.empty()) {
            std::pair<float, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            for(int exL = exLevel; exL >= std::max(0, exLevel - extendLevel); exL--) {
                int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int *) get_linklist0(curNodeNum, exL);
                } else {
                    data = (int *) get_linklist(curNodeNum, layer, exL);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *) data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visitedTag[candidate_id] == tag) continue;
                    visitedTag[candidate_id] = tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (top_candidates.size() > ef_construction)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        return top_candidates;
    }

    tableint mutuallyConnectNewElement(
            const void *data_point,
            tableint cur_c,
            std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, CompareByFirst> &top_candidates,
            int level,
            int exLevel) {
        size_t Mcurmax = level ? nM : nM0;
        getNeighborsByHeuristic2(top_candidates, nM);

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(nM);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c, exLevel);
            else
                ll_cur = get_linklist(cur_c, level, exLevel);

            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx],exLevel);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level,exLevel);

            size_t sz_link_list_other = getListCount(ll_other);


            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    float d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                             dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                }
            }
        }

        return next_closest_entry_point;
    }

    tableint addPoint(int internalId, int inlevel) {
        tableint cur_c = internalId;

        int curlevel = inlevel;

        for(int exL = 0; exL <= maxLevel; exL ++) {
            node * nd = belongTo[internalId][exL];
            int maxlevelcopy = nd->maxInternalLevel;
            tableint currObj = nd->entryPoint;
            tableint enterpoint_copy = nd->entryPoint;

            void *data_point = getDataByInternalId(internalId);

            if ((signed) currObj != -1) {
                if (curlevel < maxlevelcopy) {
                    float curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {
                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            data = get_linklist(currObj, level, exL);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                float d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {

                    std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level, exL);
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, exL);
                }
            } else {
                // Do nothing for the first element
                nd->entryPoint = internalId;
                nd->maxInternalLevel = curlevel;
            }

            if (curlevel > maxlevelcopy) {
                nd->entryPoint = cur_c;
                nd->maxInternalLevel  = curlevel;
            }
        }
        return cur_c;
    }


    tableint
    findEntry(const void *query_data,node *nd) const {
        tableint currObj = nd->entryPoint;
        float curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);

        for (int level = nd->maxInternalLevel; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level, nd->exLevel);
                int size = getListCount(data);

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    float d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        return currObj;
    }

    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst>
    searchBaseLayer0(
            tableint e1,
            tableint e2,
            int exL1,
            int exL2,
            const void *data_point,
            size_t ef,
            hnswlib::BaseFilterFunctor* isIdAllowed = nullptr) {
        tag++;

        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> candidateSet;

        float dist1 = fstdistfunc_(data_point, getDataByInternalId(e1), dist_func_param_);
        top_candidates.emplace(dist1, e1+1);
        candidateSet.emplace(-dist1, e1+1);
        visitedTag[e1] = tag;

        float dist2 = fstdistfunc_(data_point, getDataByInternalId(e2), dist_func_param_);
        top_candidates.emplace(dist2, -e2);
        candidateSet.emplace(-dist2, -e2);
        visitedTag[e2] = tag;

        float lowerBound = std::min(dist1,dist2);

        while (!candidateSet.empty()) {
            std::pair<float, tableint> curr_el_pair = candidateSet.top();
            tableint curNodeNum;
            int L;
            bool flag;
            if(curr_el_pair.second > 0){
                curNodeNum = curr_el_pair.second - 1;
                L = exL1;
                flag = 0;
            }
            else{
                curNodeNum = -curr_el_pair.second;
                L = exL2;
                flag = 1;
            }
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction) {
                break;
            }
            candidateSet.pop();

            for(int exL = L; exL >= std::max(0, L - extendLevel); exL--) {
                int *data = (int *) get_linklist0(curNodeNum, exL);

                size_t size = getListCount((linklistsizeint *) data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (vistedTag + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (vistedTag + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(h->getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(h->getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visitedTag[candidate_id] == tag) continue;
                    visitedTag[candidate_id] = tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    tableint cid;
                    if (flag)
                        cid = candidate_id + 1;
                    else
                        cid = -candidate_id;

                    float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, cid);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if ((*isIdAllowed)(candidate_id))
                            top_candidates.emplace(dist1, cid);

                        if (top_candidates.size() > ef_construction)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        return top_candidates;
    }

    linklistsizeint *get_linklist0(tableint internal_id, int exLevel) const {
        return (linklistsizeint *) (linkList_level0[internal_id] + exLevel * sizeLinkList0);
    }

    linklistsizeint *get_linklist(tableint internal_id, int level, int exLevel) const {
        return (linklistsizeint *) (linklist[internal_id] + sizeLinkList * exLevel * internalLevel[internal_id] + (level - 1) * sizeLinkList);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level, int exLevel) const {
        return level == 0 ? get_linklist0(internal_id, exLevel) : get_linklist(internal_id, level, exLevel);
    }

    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }
};


#endif //RANGEHNSW_RANGEHNSW_HPP
