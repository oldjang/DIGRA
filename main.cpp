#include <iostream>
#include "hnswlib/hnswlib.h"
#include "DataMaker.hpp"
#include "TreeHNSW.hpp"

#include <chrono>
#include <string>

#include <fstream>
#include <sstream>

#include <sys/resource.h>

int stringTonum(char *ch){
    int len = strlen(ch);
    int res = 0;
    for(int i = 0; i< len; i++){
        res = res * 10 + ch[i] - '0';
    }
    return res;
}

int main(int argc, char** argv){
    puts("read begin");
    int baseNum =stringTonum(argv[1]);
    int queryNum =stringTonum(argv[2]);
    int dim =stringTonum(argv[3]);
    int k =stringTonum(argv[4]);
    int ef_con = stringTonum(argv[5]);
    int m =stringTonum(argv[6]);
    char *baseFilename = argv[7];
    char *queryFilename = argv[8];
    char *dataFilename = argv[9];
    DataMaker dataMaker(baseFilename, queryFilename, dataFilename, baseNum, queryNum, dim);

    auto start = std::chrono::high_resolution_clock::now();
    RangeHNSW rangeHnsw(dim,baseNum,baseNum,dataMaker.data,dataMaker.key,dataMaker.value, m , ef_con);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout<<"build time:" << elapsed.count() <<std::endl;
    puts("buildOK");

    std::vector<float> r = {0.01,0.05,0.1,0.2,0.4};
    for(auto range:r){
        dataMaker.genRange(range, k);
        std::cout<<"----------"<<range<<"--------"<<std::endl;
        for(int ef = 25; ef <= 1000; ef+=25){
            float recall = 0;

            float time = 0;
            for(int i = 0 ; i < queryNum; i++){
                auto ans = dataMaker.getGt(i);

                auto start = std::chrono::high_resolution_clock::now();
                auto result = rangeHnsw.queryRange(dataMaker.query + i * dim, dataMaker.qRange[i].first,dataMaker.qRange[i].second, k, ef);
                auto end = std::chrono::high_resolution_clock::now();
                
                std::vector<int>r1, r2;
                while(!result.empty()){
                    r1.push_back(result.top().second);
                    result.pop();
                }
                for(int j = 0; j < k; j++) r2.push_back(ans[j].second);
                for(auto i1:r1)
                    for(auto i2:r2)
                        if(i1 == i2)
                            recall+=1.0/k;

                std::chrono::duration<double> elapsed = end - start;
                time += elapsed.count();
            }
            std::cout<<"ef:"<<ef<<std::endl;
            std::cout<<"recall:"<<recall/queryNum<<std::endl;
            std::cout<<"time:"<<time<<std::endl;
            std::cout<<"qps:"<<queryNum/time<<std::endl;
        }
    }

    return 0;
}
