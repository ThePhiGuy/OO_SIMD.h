#include <immintrin.h>
#include <typeinfo>
#include <stdexcept>
#include <vector>
#include <ostream>
#include <iostream>

using namespace std;

template <typename T>
class vectorRegister {
    public:
        vectorRegister();
        ~vectorRegister();
        int getRegisterSize() const;
        char getType() const;
        void zeroRegister();
        void loadRegister(const vector<T> &itemsToLoad);
        vector<T> dumpRegister() const;
        void setLoadedValues(int newVals);
        int getLoadedValues() const;

        vectorRegister<T> operator+(const vectorRegister<T>& rhs) const;
        
    private:
        alignas(64) union {
            #if defined(__SSE__)
            __m128 f128;
            #endif

            #if defined(__SSE2__)
            __m128d d128;
            __m128i i128;
            #endif

            #if defined(__AVX__)
                __m256 f256;
                __m256d d256;
            #endif

            #if defined(__AVX2__)
                __m256i i256;
            #endif

            #if defined(__AVX512F__)
                __m512 f512;
                __m512i i512;
                __m512d d512;
            #endif

        } myRegister;

        bool *supportedExtensions;
        int registerSize;
        int loadedValues;
        char type;
};

template <typename T>
ostream& operator<<(ostream& out, const vectorRegister<T> &vecReg) {
    vector<T> vecDump = vecReg.dumpRegister();
    out << "[";
    for (int i = 0; i < vecDump.size() - 1; i++) {
        out << " " << vecDump[i] << ",";
    }
    out << " " << vecDump[vecDump.size()-1] << " ]" << endl; 
    return out;
}



