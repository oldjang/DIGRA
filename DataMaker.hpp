#ifndef RANGEHNSW_DATAMAKER_HPP
#define RANGEHNSW_DATAMAKER_HPP

#include<random>
#include<vector>

#include "utils.hpp"

class DataMaker {
public:
    float *data;
    float *query;
    int *key;
    int *value;
    int *valueList;
    int baseNum, queryNum;
    int dim;
    std::vector<std::vector<std::pair<float,int> > > ans;
    std::vector<std::pair<int,int> > qRange;
    DataMaker(const char* baseFile, const char* queryFile, const char* dataFile, int N, int M, int d){
        baseNum = N;
        queryNum = M;
        dim = d;
        load_data(baseFile, data, N, dim);
        load_data(queryFile, query, M, dim);

        std::ifstream file(dataFile);
        key = new int[N];
        value = new int[N];
        valueList = new int[N];

        qRange.resize(M);
        ans.resize(M);
        for(int i = 0; i < N; i++){
            int tmp;
            file>>key[i]>>tmp;
            value[i] = tmp;
            valueList[i] = value[i];
        }
        std::sort(valueList, valueList + N);
    }
    
    DataMaker(const char* baseFile, const char* queryFile, int N, int M, int d){
        baseNum = N;
        queryNum = M;
        dim = d;
        load_data(baseFile, data, N, dim);
        load_data(queryFile, query, M, dim);

        key = new int[N];
        value = new int[N];
        valueList = new int[N];

        qRange.resize(M);
        ans.resize(M);
        
        for(int i = 0; i < N; i++){
            valueList[i] = value[i] = key[i] = i;
        }

        std::random_device rd;
        std::mt19937 g(rd());
    }

    void genRange(float range,int k){

        std::random_device rd;
        std::mt19937 rng(rd());
        int r = baseNum * range;
        std::uniform_int_distribution<int> ud(0, baseNum - r - 1);
        for(int i = 0; i< queryNum; i++){
            int L = ud(rng);
            int R = L + r;
            qRange[i] = {valueList[L],valueList[R]};
            ans[i] = getTopK(i, qRange[i].first, qRange[i].second, k);
        }

    }

    std::vector<std::pair<float,int> > getGt(int id){
        return ans[id];
    }
private:
    float getDistance(int query_id, int base_id){
        float ans = 0;
        for(int i = 0; i < dim; i++)
            ans+=(query[query_id*dim + i] - data[base_id * dim +i]) * (query[query_id*dim + i] - data[base_id * dim +i]);
        return ans;
    }

    std::vector<std::pair<float,int> > getTopK(int query_id, int L, int R, int k){
        // std::cout << query_id << ' ' << L << ' ' << R << std::endl;
        std::vector<std::pair<float,int> > result;
        for(int i = 0; i < baseNum; i++){
            if(value[i] >= L && value[i] <= R) {
                // std::cout << i << std::endl;
                result.push_back({getDistance(query_id,i),i});
            }
        }
        sort(result.begin(), result.end());
        result.resize(k);
        return result;
    }


};


#endif //RANGEHNSW_DATAMAKER_HPP
