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
#include <unistd.h>

class RangeHNSW {
public:
    RangeHNSW(
            int d,
            size_t eleNum,
            size_t maxEleNum,
            float* vecData,
            int* keyList,
            int* valueList,
            int m,
            int ef_con
    ):
            M(m),ef_construction(ef_con), space(d), dim(d), linklist(maxEleNum), searchLayer(maxEleNum), eleCount(eleNum), maxNum(maxEleNum){

        skipLayer = log(M)/log(BTREE_D);
        // M = M * 1.5;
        maxLayer = floor(log((float)maxEleNum) / log(BTREE_D));

        visited_array = new unsigned int[maxEleNum];

        std::random_device rd;  // Obtain a random number from hardware
        eng = std::mt19937 (rd());

        data_size_ = space.get_data_size();
        fstdistfunc_ = space.get_dist_func();
        dist_func_param_ = space.get_dist_func_param();

        keyList_ = new int[maxEleNum];
        valueList_ = new int[maxEleNum];
        vecData_ = (char*)(new float[maxEleNum * dim]);
        isDeleted = new bool[maxEleNum];
        memset(isDeleted,0,maxEleNum);
        memcpy(keyList_,keyList, eleNum * sizeof(int));
        memcpy(valueList_,valueList, eleNum * sizeof(int));
        memcpy(vecData_,vecData, eleNum * dim * sizeof(float));


        mult_ = 1 / log(1.0 * M);
        revSize_ = 1.0 / mult_;

        sizeLinkList = (M * sizeof(tableint) + sizeof(linklistsizeint));

        space = hnswlib::L2Space(dim);

        sortedArray.reserve(eleNum);


        for(int i = 0; i < eleNum; i++){
            key2Id[keyList[i]] = i;
            linklist[i] = (char *) malloc( (maxLayer + 1) * sizeLinkList);
            sortedArray.push_back(i);
        }

        sort(sortedArray.begin(),sortedArray.end(),[this](int a, int b) { return this->cmp(a, b); });

        root = buildTree(eleNum);

    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> queryRange(float *vecData, int rangeL, int rangeR, int k,int ef_s){
        node* highNode = findHighNode(root,rangeL,rangeR);

        int belongL = highNode->keynum;
        int belongR = highNode->keynum;
        for(int i = 0 ; i < highNode->keynum; i++){
            if(rangeL < valueList_[highNode->key[i]] ||
               (rangeL == valueList_[highNode->key[i]] && (rangeL == valueList_[findRight(highNode->child[i])]))){
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
        if(belongL == belongR) {
            top.push({0,keyList_[highNode->entryPoint]});
            return top;
        }
        int sp;
        if(belongL == belongR - 1){
            node* nodeL = highNode->child[belongL];
            while(nodeL->layer != 0 && valueList_[nodeL->key[nodeL->keynum - 1]] < rangeL) nodeL = nodeL->child[nodeL->keynum];
            tableint ep1 = nodeL->layer != 0 ? findEntry(vecData,nodeL,nodeL->child[nodeL->keynum]->entryPoint) : nodeL->entryPoint;
            ep_ids.push_back(ep1);
            searchLayer[ep1] = nodeL->layer;
            sp = highNode->key[belongL];

            node* nodeR = highNode->child[belongR];
            while(nodeR->layer != 0 && valueList_[nodeR->key[0]] > rangeR) nodeR = nodeR->child[0];
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

    void addPoint(int key,int value, char* data){
        keyList_[eleCount] = key;
        key2Id[key] = eleCount;
        valueList_[eleCount] = value;
        memcpy(vecData_+ dim * sizeof(float) * eleCount, data, dim * sizeof(float));
        linklist[eleCount] = (char *) malloc( (maxLayer + 1) * sizeLinkList);
        for(int i = 0; i <= maxLayer; i++){
            unsigned int *newListData = (unsigned int *) get_linklist(eleCount, i);

            setListCount(newListData, 0);
        }
        eleCount ++;
        addPoint(eleCount - 1);
    }

    void erase(int key){
        int id = key2Id[key];
        isDeleted[id] = true;
        erase(root,id);
        if(root->keynum == 0) root = root->child[0];
    }

    void resize(size_t newMaxN){
        int maxEleNum = newMaxN;
        skipLayer = log(M)/log(BTREE_D);

        maxLayer = floor(log((float)maxEleNum) / log(BTREE_D));

        visited_array = new unsigned int[maxEleNum];

        std::random_device rd;  // Obtain a random number from hardware
        eng = std::mt19937 (rd());

        data_size_ = space.get_data_size();
        fstdistfunc_ = space.get_dist_func();
        dist_func_param_ = space.get_dist_func_param();

        keyList_ = (int*) realloc(keyList_, maxEleNum * sizeof(int));
        valueList_ = (int*) realloc(valueList_, maxEleNum * sizeof(int));
        vecData_ = (char*)realloc(vecData_,maxEleNum * dim * sizeof(float ));
        isDeleted = (bool*) realloc(isDeleted, maxEleNum * sizeof(bool));
        memset(isDeleted,0,maxEleNum);


        mult_ = 1 / log(1.0 * M);
        revSize_ = 1.0 / mult_;

        sizeLinkList = (M * sizeof(tableint) + sizeof(linklistsizeint));

        space = hnswlib::L2Space(dim);
        linklist.resize(maxEleNum);

        for(int i = 0; i<maxEleNum;i++){
            linklist[i] = (char *) realloc(linklist[i], (maxLayer + 1) * sizeLinkList);
        }

        for(int i = maxNum; i < maxEleNum; i++){
            linklist[i] = (char *) malloc( (maxLayer + 1) * sizeLinkList);
        }
    }

private:

    size_t maxNum, eleCount;
    size_t sizeLinkList;
    struct node{
        int entryPoint = -1;
        int keynum = 0;
        int key[BTREE_M];
        struct node* child[BTREE_M + 1];
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
    char* vecData_;
    int* keyList_;
    int* valueList_;
    bool* isDeleted;
    int threshold;
    int M,M0;
    int skipLayer = 1;
    int ef_construction;
    int dim;

    int numEdges = 0;


    float alpha;
    double mult_{0.0}, revSize_{0.0};

    int maxLayer;

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

    int findRight(node* nd){
        if(nd->layer == 0) return nd->entryPoint;
        else return findRight(nd->child[nd->keynum]);
    }

    void updateEntry(node *nd){
        std::uniform_int_distribution<> distr(0,  nd->keynum);
        nd->entryPoint = nd->child[distr(eng)]->entryPoint;
    }

    node* buildTree(int eleNum){
        std::queue<std::pair< std::pair<int,int>, node* > > q[2];
        int qid = 0;
        for(int i = 0; i < eleNum; i++){
            node *nd = new node();
            nd->layer = 0;
            nd->entryPoint = sortedArray[i];
            q[qid].push({{i, i}, nd});
            unsigned int *newListData = (unsigned int *) get_linklist(sortedArray[i], 0);

            setListCount(newListData, 0);

        }
        while(q[qid].size() > 1){
            std::cout<<"layer:"<<q[qid].front().second->layer<<std::endl;
            int nxtqid = qid ^ 1;
            while(!q[qid].empty()){
                std::vector<std::pair<int,int>> tmp;
                // int numChild = (q[qid].size() >= 2 * BTREE_D) ? BTREE_D : q[qid].size();

                int numChild;
                if(q[nxtqid].size()%2 == 0){
                    numChild = (q[qid].size() >= 2 * BTREE_D) ? BTREE_D : q[qid].size();
                }
                else {
                    if (q[qid].size() >= 2 * BTREE_M) numChild = BTREE_M;
                    else if (q[qid].size() <= BTREE_M) numChild = q[qid].size();
                    else numChild = q[qid].size() / 2;
                }
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
                        unsigned int *listData = (unsigned int *) get_linklist(id, layer - 1);
                        int size = getListCount(listData);

                        tableint *listD = (tableint *) (listData + 1);
                        for (int j = 0; j < size; j++) {
                            candidates.emplace(
                                    fstdistfunc_(data, getDataByInternalId(listD[j]),
                                                 dist_func_param_), listD[j]);
                        }
                        for (int j = 0; j < numChild; j++)
                            if (i != j) {
                                tableint ep_id = findEntry(data, nd->child[j], nd->child[j]->entryPoint);
                                std::vector<tableint >ep_ids = {ep_id};
                                ResultHeap r = searchBaseLayer(ep_ids, data, layer - 1);
                                getNeighborsByHeuristic2(r, M);
                                while (!r.empty()) {
                                    auto pr = r.top();
                                    r.pop();
                                    candidates.push(pr);
                                }
                            }
                        getNeighborsByHeuristic2(candidates, M);

                        unsigned int *newListData = (unsigned int *) get_linklist(id, layer);

                        tableint *newListD = (tableint *) (newListData + 1);
                        int indx = 0;
                        while (candidates.size() > 0) {
                            newListD[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        //    std::cout<<id<<" in layer "<<nd->layer<<" has "<<indx<<" edges"<<std::endl;

                        setListCount(newListData, indx);
                        numEdges += indx;
                    }
                }
                q[nxtqid].push({{tmp[0].first,tmp[tmp.size() - 1].second}, nd});
            }

            qid = nxtqid;
        }
        std::cout<<"edge num:"<<numEdges<<std::endl;
        std::cout<<"average edges:"<<numEdges*1.0/eleNum<<std::endl;
        return q[qid].front().second;
    }

    void traverse(std::vector<tableint > &result,node* nd)
    {
        if (nd->layer == 0){
            result.push_back(nd->entryPoint);
            return;
        }
        for(int i = 0; i <= nd->keynum; i++) traverse(result,nd->child[i]);
    }


    void addPoint(int id){
        if (root == NULL)
        {
            // Allocate memory for root
            root = new node();
            root->keynum = 1;  // Update number of keys in root
        }
        else // If tree is not empty
        {
            insert(root, id);
            if(root->keynum == BTREE_M){
                node *newRoot = new node();
                newRoot->layer = root->layer + 1;
                newRoot->keynum = 0;
                newRoot->child[0] = root;
                splitNode(newRoot,0);
                for(int i = 0; i < eleCount; i++){
                    memcpy(linklist[i] + root->layer *sizeLinkList, linklist[i] + (root->layer - 1) *sizeLinkList, sizeLinkList);
                }
//                refresh(newRoot);
                root = newRoot;
            }
        }
    }



    void erase(node *nd, int id){
        int belong = nd->keynum;
        for(int i = 0 ; i < nd->keynum; i++){
            if( cmp(id, nd->key[i])){
                belong = i;
                break;
            }
        }
        if(nd->layer == 1){

            for(int i = std::max(0,belong -1); i < nd->keynum - 1; i ++){
                nd->key[i] = nd->key[i + 1];
            }
            for(int i = belong; i <= nd->keynum - 1; i ++){
                nd->child[i] = nd->child[i +1];
            }
            nd->keynum --;
            return;
        }
        else {
            erase(nd->child[belong], id);
            if (nd->child[belong]->keynum < BTREE_D - 1) {
                node *nd1 = nd->child[belong];
                if(belong > 0 && nd->child[belong - 1]->keynum >= BTREE_D){
                    node *nd2 = nd->child[belong-1];
                    for(int i = nd1->keynum - 1; i >= 0; i--){
                        nd1->key[i + 1] = nd1->key[i];
                    }
                    for(int i = nd1->keynum; i >= 0; i--){
                        nd1->child[i + 1] = nd1->child[i];
                    }
                    nd1->key[0] = nd->key[belong - 1];

                    nd1->child[0] = nd2->child[nd2->keynum];
                    nd1->keynum ++;

                    nd->key[belong - 1] = nd2->key[nd2->keynum - 1];
                    nd2->keynum --;

                    refresh(nd1, 0);
                    refresh(nd2);
                }
                else if(belong < nd->keynum && nd->child[belong + 1]->keynum >= BTREE_D){
                    node *nd2 = nd->child[belong + 1];

                    nd1->key[nd1->keynum] = nd->key[belong];
                    nd1->keynum ++;
                    nd1->child[nd1->keynum] = nd2->child[0];

                    nd->key[belong] = nd2->key[0];
                    for(int i = 0 ; i < nd2->keynum ; i++){
                        nd2->key[i] = nd2->key[i + 1];
                    }
                    for(int i = 0 ; i <= nd2->keynum ; i++){
                        nd2->child[i] = nd2->child[i + 1];
                    }
                    nd2->keynum --;

                    refresh(nd1, nd1->keynum);

                    refresh(nd2);
                }
                else if(belong > 0){
                    mergeNode(nd, belong - 1);
                }
                else{
                    mergeNode(nd, belong);
                }
            }
        }
    }

    void refresh(node *nd, int refreshId){
        std::vector<tableint> tmp;
        traverse(tmp,nd);
        int layer = nd->layer;
        int belong = 0;
        for(int i = 0 ; i < tmp.size(); i++) {
            tableint id = tmp[i];
            if (belong < nd->keynum && (!cmp(id, nd->key[belong]))) belong++;
            ResultHeap candidates;
            char *data = getDataByInternalId(id);

            if (belong == refreshId) {
                unsigned int *listData = (unsigned int *) get_linklist(id, layer - 1);
                int size = getListCount(listData);

                tableint *listD = (tableint *) (listData + 1);
                for (int j = 0; j < size; j++) {
                    if (!isDeleted[listD[j]])
                        candidates.emplace(
                                fstdistfunc_(data, getDataByInternalId(listD[j]),
                                             dist_func_param_), listD[j]);
                }

                for (int j = 0; j <= nd->keynum; j++)
                    if (j != belong) {

                        std::vector<tableint> ep_ids;
                        if (ep_ids.size() == 0) {
                            tableint ep_id = findEntry(data, nd->child[j], nd->child[j]->entryPoint);
                            ep_ids.push_back(ep_id);
                        }
                        ResultHeap r = searchBaseLayer(ep_ids, data, layer - 1);
                        getNeighborsByHeuristic2(r, M);
                        while (!r.empty()) {
                            auto pr = r.top();
                            r.pop();
                            candidates.push(pr);
                        }
                    }
            }
            else{
                unsigned int *listData = (unsigned int *) get_linklist(id, layer);
                int size = getListCount(listData);

                tableint *listD = (tableint *) (listData + 1);
                for (int j = 0; j < size; j++) {
                    if (!isDeleted[listD[j]])
                        candidates.emplace(
                                fstdistfunc_(data, getDataByInternalId(listD[j]),
                                             dist_func_param_), listD[j]);
                }

                std::vector<tableint> ep_ids;
                if (ep_ids.size() == 0) {
                    tableint ep_id = findEntry(data, nd->child[refreshId], nd->child[refreshId]->entryPoint);
                    ep_ids.push_back(ep_id);
                }
                ResultHeap r = searchBaseLayer(ep_ids, data, layer - 1);
                getNeighborsByHeuristic2(r, M);
                while (!r.empty()) {
                    auto pr = r.top();
                    r.pop();
                    candidates.push(pr);
                }
            }
            getNeighborsByHeuristic2(candidates, M);
            unsigned int *newListData = (unsigned int *) get_linklist(id, layer);

            tableint *newListD = (tableint *) (newListData + 1);
            int indx = 0;
            while (candidates.size() > 0) {
                newListD[indx] = candidates.top().second;
                candidates.pop();
                indx++;
            }
            setListCount(newListData, indx);
        }
        updateEntry(nd);
    }

    void mergeNode(node *nd, int mergeId){
        node* n1 = nd->child[mergeId];
        node* n2 = nd->child[mergeId + 1];
        n1->key[n1->keynum] = nd->key[mergeId];
        for(int i = 0; i < n2->keynum; i++){
            n1->key[i + n1->keynum + 1] = n2->key[i];
        }
        for(int i = 0; i <= n2->keynum; i++){
            n1->child[i + n1->keynum + 1] = n2->child[i];
        }
        n1->keynum += n2->keynum + 1;
        refresh(n1);
        for(int i = mergeId; i < nd->keynum; i++){
            nd->key[i] = nd->key[i + 1];
        }
        for(int i = mergeId + 1; i <= nd->keynum; i++){
            nd->child[i] = nd->child[i + 1];
        }
        nd->keynum --;
        updateEntry(nd);
    }

    tableint insert(node* nd, int id){
        int belong = nd->keynum;
        for(int i = 0 ; i < nd->keynum; i++){
            if( cmp(id, nd->key[i])){
                belong = i;
                break;
            }
        }
        tableint ep_id;
        if(nd->layer == 1){
            node *newnd = new node();
            newnd->layer = 0;
            newnd->entryPoint = id;

            unsigned int *newListData = (unsigned int *) get_linklist(id, 1);
            tableint *newListD = (tableint *) (newListData + 1);
            int indx = 0;
            for(int i = 0; i <=nd->keynum; i++){
                newListD[indx] = nd->child[i]->entryPoint;
                indx++;
            }
            setListCount(newListData, indx);

            for(int i = nd->keynum -1; i >= belong; i --){
                nd->key[i + 1] = nd->key[i];
            }
            for(int i = nd->keynum; i > belong; i --){
                nd->child[i + 1] = nd->child[i];
            }
            if(belong == 0){
                if(cmp(id,nd->child[0]->entryPoint)){
                    std::swap(nd->child[0],newnd);
                }
            }
            nd->key[belong] = newnd->entryPoint;
            nd->keynum ++;
            nd->child[belong + 1] = newnd;
            ep_id = nd->entryPoint;
        }
        else {
            ep_id = insert(nd->child[belong], id);
            if (nd->child[belong]->keynum == BTREE_M) {
                splitNode(nd, belong);
            }
        }
        std::vector<tableint >ep_ids = {ep_id};
        char *data = getDataByInternalId(id);
        auto candidates = searchBaseLayer(ep_ids, data,nd->layer);
        getNeighborsByHeuristic2(candidates,M);
        return connectEdges(data,id,candidates, nd->layer);
    }

    void splitNode(node *nd, int splitId){
        node* n1 = nd->child[splitId];
        node* n2 = new node();
        n2->layer = n1->layer;
        n2->keynum = 0;
        int splitPoint = n1->keynum/2;
        int newKey = n1->key[splitPoint];
        for(int i = splitPoint + 1; i < n1->keynum; i++){
            n2->key[i-splitPoint - 1] = n1->key[i];
        }
        for(int i = splitPoint + 1; i <= n1->keynum; i++){
            n2->child[i-splitPoint - 1] = n1->child[i];
        }
        n2->keynum = n1->keynum - splitPoint - 1;
        n1->keynum = splitPoint;
        refresh(n1);
        refresh(n2);
        for(int i = nd->keynum -1; i >= splitId; i --){
            nd->key[i + 1] = nd->key[i];
        }
        for(int i = nd->keynum; i > splitId; i --){
            nd->child[i + 1] = nd->child[i];
        }
        nd->keynum ++;
        nd->key[splitId] = newKey;
        nd->child[splitId + 1] = n2;
        updateEntry(nd);
    }


    void refresh(node *nd){
        std::vector<tableint> tmp;
        traverse(tmp,nd);
        int layer = nd->layer;
        int belong = 0;
        for(int i = 0 ; i < tmp.size(); i++){
            tableint id = tmp[i];
            if(belong<nd->keynum&&(!cmp(id,nd->key[belong]))) belong++;
            ResultHeap candidates;
            char *data = getDataByInternalId(id);

            unsigned int *prelistData = (unsigned int *) get_linklist(id, layer);
            int presize = getListCount(prelistData);

            tableint *prelistD = (tableint *) (prelistData + 1);


            unsigned int *listData = (unsigned int *) get_linklist(id, layer - 1);
            int size = getListCount(listData);

            tableint *listD = (tableint *) (listData + 1);
            for (int j = 0; j < size; j++) {
                if(!isDeleted[listD[j]])
                    candidates.emplace(
                            fstdistfunc_(data, getDataByInternalId(listD[j]),
                                         dist_func_param_), listD[j]);
            }

            for(int j = 0; j <= nd->keynum; j++)
                if(j!=belong){

                    std::vector<tableint >ep_ids;
                    for (int k = 0; k < presize; k++) {
                        if(!isDeleted[prelistD[k]])
                            if((j == 0 && (!cmp(prelistD[k],tmp[0]))|| ((j!=0)&&(!cmp(prelistD[k],nd->key[j - 1])))))
                                if((j == nd->keynum && cmp(prelistD[k], tmp[tmp.size() - 1]))|| ((j!=nd->keynum)&&cmp(prelistD[k], nd->key[j]))){
                                    ep_ids.push_back(prelistD[k]);
                                }
                    }
                    if(ep_ids.size() == 0) {
                        tableint ep_id = findEntry(data, nd->child[j], nd->child[j]->entryPoint);
                        ep_ids.push_back(ep_id);
                    }
                    ResultHeap r = searchBaseLayer(ep_ids, data, layer - 1);
                    getNeighborsByHeuristic2(r, M);
                    while (!r.empty()) {
                        auto pr = r.top();
                        r.pop();
                        candidates.push(pr);
                    }
                }
            getNeighborsByHeuristic2(candidates, M);

            unsigned int *newListData = (unsigned int *) get_linklist(id, layer);

            tableint *newListD = (tableint *) (newListData + 1);
            int indx = 0;
            while (candidates.size() > 0) {
                newListD[indx] = candidates.top().second;
                candidates.pop();
                indx++;
            }
            setListCount(newListData, indx);
        }
        updateEntry(nd);
    }

    tableint connectEdges(
            const void *data_point,
            tableint cur_c,
            ResultHeap &top_candidates,
            int layer) {

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            linklistsizeint *ll_cur = get_linklist(cur_c, layer);

            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

            linklistsizeint *ll_other = get_linklist(selectedNeighbors[idx], layer);

            size_t sz_link_list_other = getListCount(ll_other);

            tableint *data = (tableint *) (ll_other + 1);
            if (sz_link_list_other < M) {
                data[sz_link_list_other] = cur_c;
                setListCount(ll_other, sz_link_list_other + 1);
            } else {
                // finding the "weakest" element to replace it with the new one
                float d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                           dist_func_param_);
                // Heuristic:
                ResultHeap candidates;
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(
                            fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                         dist_func_param_), data[j]);
                }

                getNeighborsByHeuristic2(candidates, M);

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }

                setListCount(ll_other, indx);
            }
        }

        return next_closest_entry_point;
    }

    node* findHighNode(node* node,int rangeL, int rangeR){
        if(node->layer == 0){
            return node;
        }
        int belongL = node->keynum;
        int belongR = node->keynum;
        for(int i = 0 ; i < node->keynum; i++){
            if(rangeL < valueList_[node->key[i]] ||
               (rangeL == valueList_[node->key[i]] && (rangeL == valueList_[findRight(node->child[i])]))){
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
        return (char*)(vecData_ + internal_id * data_size_);
    }

    ResultHeap searchBaseLayer(const std::vector<tableint> &ep_ids, const void *data_point, int layer) {
        tag ++;

        ResultHeap top_candidates;
        ResultHeap candidateSet;

        float lowerBound;

        for(int i = 0; i < ep_ids.size(); i++) {
            int ep_id = ep_ids[i];
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            if(!isDeleted[ep_id]) {
                top_candidates.emplace(dist, ep_id);
                candidateSet.emplace(-dist, ep_id);
            }
            else{
                candidateSet.emplace(-std::numeric_limits<float>::max(), ep_id);
            }
            visited_array[ep_id] = tag;
        }

        if(!top_candidates.empty())
            lowerBound = top_candidates.top().first;
        else
            lowerBound = std::numeric_limits<float>::max();


        while (!candidateSet.empty()) {
            std::pair<float, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            for(int i = 0; i <= 0; i++) {
                if(layer - i <= 0) break;
                int *data = (int *) get_linklist(curNodeNum, layer - i);
                size_t size = getListCount((linklistsizeint *) data);
                tableint *datal = (tableint *) (data + 1);

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + *(datal + j + 2)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 2)), _MM_HINT_T0);
                    // _mm_prefetch((char *) (visited_array + *(datal + j + 3)), _MM_HINT_T0);
                    // _mm_prefetch(getDataByInternalId(*(datal + j + 3)), _MM_HINT_T0);
                    // _mm_prefetch((char *) (visited_array + *(datal + j + 4)), _MM_HINT_T0);
                    // _mm_prefetch(getDataByInternalId(*(datal + j + 4)), _MM_HINT_T0);
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

                        if(!isDeleted[candidate_id])
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

        float lowerBound;
        for(int i = 0; i < ep_ids.size(); i++) {
            int ep_id = ep_ids[i];
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            if(!isDeleted[ep_id] && valueList_[ep_id]>=rangeL && valueList_[ep_id] <= rangeR) {
                top_candidates.emplace(dist, ep_id);
                candidateSet.emplace(-dist, ep_id);
            }
            else{
                candidateSet.emplace(-std::numeric_limits<float>::max(), ep_id);
            }
            visited_array[ep_id] = tag;
        }

        if(!top_candidates.empty())
            lowerBound = top_candidates.top().first;
        else
            lowerBound = std::numeric_limits<float>::max();

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
                int *data = (int *) get_linklist(curNodeNum, layer - i);

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
#ifdef USE_SSE
                    // if(j%2 == 0){
                        _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                        _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + *(datal + j + 2)), _MM_HINT_T0);
                        _mm_prefetch(getDataByInternalId(*(datal + j + 2)), _MM_HINT_T0);
                    // }
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

                        if(!isDeleted[candidate_id])
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
                int *data = (int *) get_linklist(curNodeNum, Layer);

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
#ifdef USE_SSE
                    // if(j%2==0){
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + *(datal + j + 2)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 2)), _MM_HINT_T0);
                    // }
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
                unsigned int *data;

                for(int l = 0; l <= 0 ; l++){
                    data = (unsigned int *) get_linklist(currObj, layer-l);
                    int size = getListCount(data);

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
                        _mm_prefetch(getDataByInternalId(*(datal + i + 2)), _MM_HINT_T0);
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
        }
        return currObj;
    }


    linklistsizeint *get_linklist(tableint internal_id, int layer) const {
        return (linklistsizeint *) (linklist[internal_id] + sizeLinkList * layer);
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }

    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }
};


#endif //RANGEHNSW_RANGEHNSW_HPP
