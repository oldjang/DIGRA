#ifndef RANGEHNSW_RANGEHNSW_HPP
#define RANGEHNSW_RANGEHNSW_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
//#include <sys/resource.h>

//#include "hnswlib/hnswlib.h"
#include "HNSW.hpp"
#define BTREE_M 3
#define BTREE_D 2


using namespace hnswlib;

#include <sys/resource.h>

void printMemoryUsage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        std::cout << "Max Resident Set Size: " << usage.ru_maxrss << " KB\n";
    } else {
        std::cerr << "Error getting resource usage\n";
    }
}

class RangeHNSW {
public:
    RangeHNSW(
            int d,
            size_t eleNum,
            char* vecData,
            int* keyList,
            int* valueList,
            int m,
            int ef_con
    ):
            M(m),ef_construction(ef_con), space(d), dim(d), linklist(eleNum), searchLayer(eleNum){

        M0 = M * 2;
        skipLayer = log(M)/log(BTREE_D);
        // M = M * 1.5;
        maxLayer = floor(log((float)eleNum) / log(BTREE_D));

        visited_array = new unsigned int[eleNum];

        std::random_device rd;  // Obtain a random number from hardware
        eng = std::mt19937 (rd());

        data_size_ = space.get_data_size();
        fstdistfunc_ = space.get_dist_func();
        dist_func_param_ = space.get_dist_func_param();

        keyList_ = std::vector<int>(keyList, keyList + eleNum);
        valueList_ = std::vector<int>(valueList, valueList + eleNum);
        vecData_.resize(eleNum);
        int vectorSize = dim * sizeof(float);
        for(int i = 0; i < eleNum; i++)
        {
            vecData_[i] = new char[vectorSize];
//            for(int j = 0; j< vectorSize; j++)vecData_[i][j] = (vecData+i*vectorSize)[j];
            memcpy(vecData_[i], vecData + i * vectorSize,  vectorSize);
//            memcpy(vecData_[i], vecData + i,  vectorSize);
        }

        printMemoryUsage();

        mult_ = 1 / log(1.0 * M);
        revSize_ = 1.0 / mult_;

//        sizeLinkList = (M * sizeof(tableint) + sizeof(linklistsizeint));

        space = hnswlib::L2Space(dim);

        sortedArray.reserve(eleNum);

        edges.resize(eleNum);


        for(int i = 0; i < eleNum; i++){
            key2Id[keyList[i]] = i;
            edges[i].resize(maxLayer + 1);
//            linklist[i] = (char *) malloc( (maxLayer + 1) * sizeLinkList);
            sortedArray.push_back(i);
        }

        sort(sortedArray.begin(),sortedArray.end(),[this](int a, int b) { return this->cmp(a, b); });

        printMemoryUsage();
        root = buildTree(eleNum);

        printMemoryUsage();

    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> queryRange(float *vecData, int rangeL, int rangeR, int k,int ef_s){
        node* highNode = findHighNode(root,rangeL,rangeR);

        int belongL = highNode->keynum;
        int belongR = highNode->keynum;
        for(int i = 0 ; i < highNode->keynum; i++){
            if(rangeL <= valueList_[highNode->key[i]]){
                belongL = i;
                break;
            }
        }
        for(int i = 0 ; i < highNode->keynum; i++){
            if(rangeR < valueList_[highNode->key[i]]){
                belongR = i;
                break;
            }
        }
        std::vector<tableint > ep_ids;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> top;
        if(belongL == belongR) return top;
        int sp;
        if(belongL == belongR - 1){
            node* nodeL = highNode->child[belongL];
            while(nodeL->layer != 0 && nodeL->key[nodeL->keynum - 1] <= rangeL) nodeL = nodeL->child[nodeL->keynum];
            tableint ep1 = nodeL->layer != 0 ? findEntry(vecData,nodeL,nodeL->child[nodeL->keynum]->entryPoint) : nodeL->entryPoint;
            ep_ids.push_back(ep1);
            searchLayer[ep1] = nodeL->layer;
            sp = highNode->key[belongL];

            node* nodeR = highNode->child[belongR];
            while(nodeR->layer != 0 && nodeR->key[0] > rangeR) nodeR = nodeR->child[0];
            tableint ep2 = nodeR->layer != 0 ? findEntry(vecData,nodeR,nodeR->child[0]->entryPoint) : nodeR->entryPoint;
            ep_ids.push_back(ep2);
            searchLayer[ep2] = nodeR->layer;
        }
        else{
            sp = -1;
            std::uniform_int_distribution<> distr(belongL + 1, belongR -1);
            tableint high_ep = highNode->child[distr(eng)]->entryPoint;
            tableint ep = findEntry(vecData,highNode, high_ep);
            ep_ids.push_back(ep);
            searchLayer[ep] = highNode->layer;
        }
        ResultHeap result = searchBaseLayer0(ep_ids,vecData,highNode->layer,rangeL,rangeR,ef_s,sp);

        while(result.size() > k) result.pop();

        while(!result.empty()){
            auto r = result.top();
            result.pop();
            top.push({r.first,keyList_[r.second]});
        }
        return top;
    }

private:

    size_t sizeLinkList;
    struct node{
        int entryPoint = -1;
        int keynum = BTREE_D - 1;
        int key[BTREE_M-1];
        struct node* child[BTREE_M];
        short int layer; //layer in tree

        node(){}

    };

    bool cmp(int a,int b){
        if(valueList_[a]!=valueList_[b]) return valueList_[a]<valueList_[b];
        else return keyList_[a]<keyList_[b];
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<float, tableint> const& a,
                                  std::pair<float, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    typedef std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float , tableint>>, CompareByFirst> ResultHeap;

    hnswlib::L2Space space;
    size_t data_size_{0};

    DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_{nullptr};

    node* root;
    std::vector<char* > vecData_;
    std::vector<int> keyList_;
    std::vector<int> valueList_;
//    char* vecData_;
//    int* keyList_;
//    int* valueList_;
    int threshold;
    int M,M0;
    int skipLayer = 1;
    int ef_construction;
    int dim;


    float alpha;
    double mult_{0.0}, revSize_{0.0};

    int maxLayer;

    std::vector<std::vector<std::vector<int> > > edges;

    std::vector<char *> linklist;

    std::vector<short int> searchLayer;
    unsigned int *visited_array;
    unsigned int tag = 0;
    std::vector<int> sortedArray;

    std::unordered_map<int,int> key2Id;

    std::mt19937 eng; // Seed the generator

    int findEntryLayer(int Layer) const{
        return Layer % skipLayer;
    }


    node* buildTree(int eleNum){
        std::queue<std::pair< std::pair<int,int>, node* > > q[2];
        int qid = 0;
        for(int i = 0; i < eleNum; i++){
            node *nd = new node();
            nd->layer = 0;
            nd->entryPoint = sortedArray[i];
            q[qid].push({{i, i}, nd});

        }
        while(q[qid].size() > 1){
            std::cout<<"layer:"<<q[qid].front().second->layer<<std::endl;
            int nxtqid = qid ^ 1;
            while(!q[qid].empty()){
                std::vector<std::pair<int,int>> tmp;
                int numChild = (q[qid].size() >= 2 * BTREE_D) ? BTREE_D : q[qid].size();
                tmp.reserve(numChild);
                tmp.resize(numChild);
                node* nd = new node();
                nd->keynum = numChild - 1 ;
                for(int i = 0; i < numChild; i++){
                    auto t = q[qid].front();
                    tmp[i] = t.first;
                    q[qid].pop();
                    if (i != 0){
                        nd->key[i - 1] = sortedArray[tmp[i].first];
                    }
                    nd->child[i] = t.second;
                }
                std::uniform_int_distribution<> distr(0, numChild - 1);
                nd->entryPoint = nd->child[distr(eng)]->entryPoint;
                int layer = nd->layer = nd->child[0]->layer + 1;

                for(int i = 0; i < numChild; i++) {
                    for (int ii = tmp[i].first; ii <= tmp[i].second; ii++) {
                        int id = sortedArray[ii];
                        ResultHeap candidates;
                        char *data = getDataByInternalId(id);

                        auto &edgeList = edges[id][layer - 1];
                        int size = edgeList.size();

                        for (int j = 0; j < size; j++) {
                            candidates.emplace(
                                    fstdistfunc_(data, getDataByInternalId(edgeList[j]),
                                                 dist_func_param_), edgeList[j]);
                        }
                        for (int j = 0; j < numChild; j++)
                            if (i != j) {
                                tableint ep_id = findEntry(data, nd->child[j], nd->child[j]->entryPoint);
                                ResultHeap r = searchBaseLayer(ep_id, data, layer - 1);
                                getNeighborsByHeuristic2(r, M);
                                while (!r.empty()) {
                                    auto pr = r.top();
                                    r.pop();
                                    candidates.push(pr);
                                }
                            }
                        getNeighborsByHeuristic2(candidates, M);


                        auto &newEdgeList = edges[id][layer];
                        newEdgeList.resize(candidates.size());

                        int indx = 0;
                        while (candidates.size() > 0) {
                            newEdgeList[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }
                    }
                }
                q[nxtqid].push({{tmp[0].first,tmp[tmp.size() - 1].second}, nd});
            }

            qid = nxtqid;
        }
        return q[qid].front().second;
    }


    node* findHighNode(node* node,int rangeL, int rangeR){
        if(node->layer == 0){
            return node;
        }
        int belongL = node->keynum;
        int belongR = node->keynum;
        for(int i = 0 ; i < node->keynum; i++){
            if(rangeL <= valueList_[node->key[i]]){
                belongL = i;
                break;
            }
        }
        for(int i = 0 ; i < node->keynum; i++){
            if(rangeR < valueList_[node->key[i]]){
                belongR = i;
                break;
            }
        }
        if(belongL == belongR) return findHighNode(node->child[belongL], rangeL, rangeR);
        return node;
    }


    void getNeighborsByHeuristic2(
            ResultHeap &top_candidates,
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
        return vecData_[internal_id];
    }

    ResultHeap searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        tag ++;

        ResultHeap top_candidates;
        ResultHeap candidateSet;

        float lowerBound;

        float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidateSet.emplace(-dist, ep_id);

        visited_array[ep_id] = tag;

        while (!candidateSet.empty()) {
            std::pair<float, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;


            for(int i = 0; i <= 1; i++) {
                if(layer - i <= 0) break;
                auto &edgeList = edges[curNodeNum][layer - i];
                size_t size = edgeList.size();

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = edgeList[j];
#ifdef USE_SSE
                    if (j < size - 1){
                        _mm_prefetch((char *) (visited_array + (edgeList[j + 1])), _MM_HINT_T0);
                        _mm_prefetch(getDataByInternalId((edgeList[j + 1])), _MM_HINT_T0);
                    }
#endif
                    if (visited_array[candidate_id] == tag) continue;
                    visited_array[candidate_id] = tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        top_candidates.emplace(dist1, candidate_id);
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

    ResultHeap
    searchBaseLayer0(std::vector<tableint> ep_ids, const void *data_point, int Layer, int rangeL, int rangeR, int ef, int splitPoint) {
        tag ++;

        ResultHeap top_candidates;
        ResultHeap candidateSet;
        for (tableint ep_id:ep_ids) {
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            candidateSet.emplace(-dist, ep_id);
        }

        float lowerBound = -candidateSet.top().first;

        while (!candidateSet.empty()) {
            std::pair<float, tableint> curr_el_pair = candidateSet.top();
            tableint curNodeNum = curr_el_pair.second;
            short int layer = searchLayer[curNodeNum];
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef) {
                break;
            }
            candidateSet.pop();

            for(int i = 0; i <= 1; i++) {
                if(layer - i <= 0) break;

                auto &edgeList = edges[curNodeNum][layer - i];

                size_t size = edgeList.size();
#ifdef USE_SSE
                if(size > 0) _mm_prefetch((char *) (visited_array + edgeList[0]), _MM_HINT_T0);
                if(size > 0) _mm_prefetch(getDataByInternalId(edgeList[0]), _MM_HINT_T0);
                if(size > 1) _mm_prefetch(getDataByInternalId(edgeList[1]), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = edgeList[j];
#ifdef USE_SSE
                    if(j < size - 1) _mm_prefetch((char *) (visited_array + edgeList[j+1]), _MM_HINT_T0);
                    if(j < size - 1) _mm_prefetch(getDataByInternalId(edgeList[j+1]), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == tag) continue;
                    // if ( !(valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR))continue;
                    visited_array[candidate_id] = tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    tableint cid = candidate_id;

                    float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, cid);
                        searchLayer[cid] = layer;
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if ( valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR)
                            top_candidates.emplace(dist1, cid);

                        if (top_candidates.size() > ef)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }

            // if(splitPoint!=-1) {
            if(splitPoint!=-1) {
                auto &edgeList = edges[curNodeNum][Layer];

                size_t size = edgeList.size();
#ifdef USE_SSE
                if(size > 0) _mm_prefetch((char *) (visited_array + edgeList[0]), _MM_HINT_T0);
                if(size > 0) _mm_prefetch(getDataByInternalId(edgeList[0]), _MM_HINT_T0);
                if(size > 1) _mm_prefetch(getDataByInternalId(edgeList[1]), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = edgeList[j];
#ifdef USE_SSE
                    if(j < size - 1) _mm_prefetch((char *) (visited_array + edgeList[j+1]), _MM_HINT_T0);
                    if(j < size - 1) _mm_prefetch(getDataByInternalId(edgeList[j+1]), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == tag) continue;
                    if ( !(valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR))continue;
                    visited_array[candidate_id] = tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    tableint cid = candidate_id;

                    float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, cid);
                        searchLayer[cid] = searchLayer[ep_ids[0]] == layer ? searchLayer[ep_ids[1]]: searchLayer[ep_ids[0]];
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if ( valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR)
                            top_candidates.emplace(dist1, cid);

                        if (top_candidates.size() > ef)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        return top_candidates;
    }

    tableint
    findEntry(const void *query_data, node *nd, tableint currObj) const {
        float curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        int endLayer = nd->layer;
        int startLayer = findEntryLayer(endLayer);

        for (int layer = startLayer; layer < endLayer; layer += skipLayer) {
            bool changed = true;
            while (changed) {
                changed = false;

                auto &edgeList = edges[currObj][layer];

                size_t size = edgeList.size();

                for (int i = 0; i < size; i++) {
                    tableint cand = edgeList[i];
#ifdef USE_SSE
                    if(i < size - 1) _mm_prefetch(getDataByInternalId(edgeList[i+1]), _MM_HINT_T0);
#endif
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


//    linklistsizeint *get_linklist(tableint internal_id, int layer) const {
//        return (linklistsizeint *) (linklist[internal_id] + sizeLinkList * layer);
//    }
//
//
//    unsigned short int getListCount(linklistsizeint * ptr) const {
//        return *((unsigned short int *)ptr);
//    }
//
//    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
//        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
//    }
};


#endif //RANGEHNSW_RANGEHNSW_HPP
