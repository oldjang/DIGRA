#ifndef RANGEHNSW_RANGEHNSW_HPP
#define RANGEHNSW_RANGEHNSW_HPP

#include <vector>
//#include <sys/resource.h>

//#include "hnswlib/hnswlib.h"
#include "HNSW.hpp"

using namespace hnswlib;
class RangeHNSW {
public:
    RangeHNSW(
            int d,
            size_t eleNum,
            float* vecData,
            int* idList,
            int* valueList,
            int th,
            int m,
            int ef_con
            ):
            threshold(th),M(m),ef_construction(ef_con), space(d), dim(d){
        idList_ = new int[eleNum];
        valueList_ = new int[eleNum];
        vecData_ = new float[eleNum * dim];
        memcpy(idList_,idList, eleNum * sizeof(int));
        memcpy(valueList_,valueList, eleNum * sizeof(int));
        memcpy(vecData_,vecData, eleNum * dim * sizeof(float));
        root = buildTree(0,eleNum - 1);
        space = hnswlib::L2Space(dim);
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> query(float *vecData, int rangeL, int rangeR, int k,int ef_s){
        Filter *filter = new Filter(this, rangeL, rangeR);
        return queryInTree(root,rangeL, rangeR, filter, vecData, k, ef_s);
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> queryRange(float *vecData, int rangeL, int rangeR, int k,int ef_s){
        Filter *filter = new Filter(this, rangeL, rangeR);
        auto nodes = findNode(root,rangeL, rangeR);
        if(nodes.second == nullptr) return nodes.first->hnsw->searchKnn(vecData,k,filter);
        if(nodes.first == nullptr) return nodes.second->hnsw->searchKnn(vecData,k,filter);
        tableint e1 = nodes.first->hnsw->findEntry(vecData);
        tableint e2 = nodes.second->hnsw->findEntry(vecData);


    }

private:

    struct node{
        int L, R;
        hnswlib::HNSW<float> *hnsw;
        node *leftSon, *rightSon;

        node(){}
        node(int l,int r, node* ls = nullptr, node* rs = nullptr):L(l),R(r),leftSon(ls),rightSon(rs){}
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
            return (rangeHnsw->idList_[id] >= rangeL) && (rangeHnsw->idList_[id] <= rangeR);
        }
    private:
        RangeHNSW *rangeHnsw;
        int rangeL,rangeR;
    };


    hnswlib::L2Space space;
    node* root;
    float* vecData_;
    int* idList_;
    int* valueList_;
    int threshold;
    int M;
    int ef_construction;
    int dim;

    std::vector<std::pair<float, tableint> > visitedList;
    std::vector<short int > visitedTag;
    int tag = 0;




    node* buildTree(int L,int R){
        node* newNode = new node(valueList_[L],valueList_[R]);
        printf("%d %d",L,R);puts("");
        buildHNSW(newNode, L, R);
//        newNode->hnsw->countSize();
//        struct rusage usage;
//        if (getrusage(RUSAGE_SELF, &usage) == 0) {
//            std::cout << "Virtual memory used: " << usage.ru_maxrss/1024.0 << " Mb" << std::endl;
//        } else {
//            std::cerr << "Failed to get memory information." << std::endl;
//        }

        if(R - L + 1 <= threshold) return newNode;
        int mid = (L + R) / 2;
        newNode->leftSon = buildTree(L, mid);
        newNode->rightSon = buildTree(mid + 1, R);
        return newNode;
    }

    void buildHNSW(node* node, int L, int R){
        node->hnsw = new hnswlib::HNSW<float>(&space, R-L+1, M, ef_construction,(char *)vecData_);
        for(int i = L; i <= R; i++)
            node->hnsw->addPoint(vecData_ + i * dim, i);
    }

    std::pair<node*, node*> findNode(node* node,int rangeL, int rangeR){
        if(node->rightSon == nullptr || node->leftSon == nullptr){
            return {node, nullptr};
        }

        if(node->rightSon->L > rangeR)
            return findNode(node->leftSon, rangeL, rangeR);
        if(node->leftSon->R < rangeL)
            return findNode(node->rightSon, rangeL, rangeR);

        if((rangeL <= node->leftSon->L && node->leftSon->R <= rangeR)
           ||(rangeL <= node->rightSon->L && node->rightSon->R <= rangeR)) {
            return {node, nullptr};
        }
        return {findNode(node->leftSon, rangeL, rangeR).first, findNode(node->rightSon, rangeL, rangeR).first};
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> queryInTree(node* node,int rangeL, int rangeR, Filter *filter, float* vecData, int k, int ef_s){
        if(node->rightSon == nullptr || node->leftSon == nullptr){
            node->hnsw->setEf(ef_s);
//            printf("%d %d\n",node->L, node->R);
            return node->hnsw->searchKnn(vecData,k,filter);
        }

        if(node->rightSon->L > rangeR)
            return queryInTree(node->leftSon, rangeL, rangeR, filter, vecData, k, ef_s);
        if(node->leftSon->R < rangeL)
            return queryInTree(node->rightSon, rangeL, rangeR, filter, vecData, k, ef_s);

        if((rangeL <= node->leftSon->L && node->leftSon->R <= rangeR)
            ||(rangeL <= node->rightSon->L && node->rightSon->R <= rangeR)) {
            node->hnsw->setEf(ef_s);
//            printf("%d %d\n",node->L, node->R);
            return node->hnsw->searchKnn(vecData, k, filter);
        }

        auto LAns = queryInTree(node->leftSon, rangeL, rangeR, filter, vecData, k, ef_s);
        auto RAns = queryInTree(node->rightSon, rangeL, rangeR, filter, vecData, k, ef_s);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> ans;
        ans = LAns;
        while(!RAns.empty()){
            ans.push(RAns.top());
            RAns.pop();
        }
        return ans;

    }

    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst>
    searchBaseLayer(
            tableint e1,
            tableint e2,
            HNSW<float> *hnsw1,
            HNSW<float> *hnsw2,
            const void *data_point,
            size_t ef,
            hnswlib::BaseFilterFunctor* isIdAllowed = nullptr) {
        tag++;

        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst> candidateSet;

        float dist1 = hnsw1->fstdistfunc_(data_point, hnsw1->getDataByInternalId(e1), hnsw1->dist_func_param_);
        top_candidates.emplace(dist1, e1+1);
        candidateSet.emplace(-dist1, e1+1);
        visitedTag[hnsw1->getExternalLabel(e1)] = tag;

        float dist2 = hnsw2->fstdistfunc_(data_point, hnsw2->getDataByInternalId(e2), hnsw2->dist_func_param_);
        top_candidates.emplace(dist2, -e2);
        candidateSet.emplace(-dist2, -e2);
        visitedTag[hnsw1->getExternalLabel(e2)] = tag;

        float lowerBound = std::min(dist1,dist2);

        while (!candidateSet.empty()) {
            std::pair<float, tableint> curr_el_pair = candidateSet.top();
            tableint curNodeNum;
            HNSW<float> *h;
            bool flag;
            if(curr_el_pair.second > 0){
                curNodeNum = curr_el_pair.second - 1;
                h = hnsw1;
                flag = 0;
            }
            else{
                curNodeNum = -curr_el_pair.second;
                h = hnsw2;
                flag = 1;
            }
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == h->ef_construction_) {
                break;
            }
            candidateSet.pop();

            int *data = (int*)h->get_linklist0(curNodeNum);

            size_t size = h->getListCount((linklistsizeint*)data);
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
                char *currObj1 = (h->getDataByInternalId(candidate_id));

                tableint cid;
                if(flag)
                    cid = candidate_id + 1;
                else
                    cid = -candidate_id;

                float dist1 = h->fstdistfunc_(data_point, currObj1, h->dist_func_param_);
                if (top_candidates.size() < h->ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, cid);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!h->isMarkedDeleted(candidate_id) && (*isIdAllowed)(h->getExternalLabel(candidate_id)))
                        top_candidates.emplace(dist1, cid);

                    if (top_candidates.size() > h->ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }

        return top_candidates;
    }
};


#endif //RANGEHNSW_RANGEHNSW_HPP
