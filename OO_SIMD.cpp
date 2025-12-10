#include "OO_SIMD.h"

using namespace std;

template <typename T>
vectorRegister<T>::vectorRegister() {
    if (is_same<T, int>::value) {
        type = 'i';
    }
    else if (is_same<T, double>::value) {
        type = 'd';
    }
    else if (is_same<T, float>::value) {
        type = 'f';
    }
    else {
        throw invalid_argument("Vector Registers can only be of Type : int, double or float.");
    }

    supportedExtensions = new bool[5]();
    loadedValues = 0;

    #if defined(__SSE__)
        if (__builtin_cpu_supports("sse")) {
            supportedExtensions[0] = true;
        }   
        else {
            supportedExtensions[0] = false;
        }
    #endif

    #if defined(__SSE2__)
        if (__builtin_cpu_supports("sse2")) {
            supportedExtensions[1] = true;
        }   
        else {
            supportedExtensions[1] = false;
        }
    #endif

    #if defined(__AVX__)
        if (__builtin_cpu_supports("avx")) {
            supportedExtensions[2] = true;
        }   
        else {
            supportedExtensions[2] = false;
        }
    #endif

    #if defined(__AVX2__)
        if (__builtin_cpu_supports("avx2")) {
            supportedExtensions[3] = true;
        }   
        else {
            supportedExtensions[3] = false;
        }
    #endif

    #if defined(__AVX512F__)
        if (__builtin_cpu_supports("avx512f")) {
            supportedExtensions[4] = true;
        }   
        else {
            supportedExtensions[4] = false;
        }
    #endif
   

    switch(type) {
        case 'i':
            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                myRegister.i512 = _mm512_setzero_si512();
                registerSize = 16;
                break;
            }
            #endif

            #if defined(__AVX2__)
            if (supportedExtensions[3]) {
                myRegister.i256 = _mm256_setzero_si256();
                registerSize = 8;
                break;
            }
            #endif

            #if defined(__SSE2__)
            if (supportedExtensions[1]) {
                myRegister.i128 = _mm_setzero_si128();
                registerSize = 4;
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Integers.");
            break;

        case 'd':
            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                myRegister.d512 = _mm512_setzero_pd();
                registerSize = 8;
                break;
            }
            #endif

            #if defined(__AVX__)
            if (supportedExtensions[2]) {
                myRegister.d256 = _mm256_setzero_pd();
                registerSize = 4;
                break;
            }
            #endif

            #if defined(__SSE2__)
            if (supportedExtensions[1]) {
                myRegister.d128 = _mm_setzero_pd();
                registerSize = 2;
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Doubles.");
            break;

        case 'f':
            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                myRegister.f512 = _mm512_setzero_ps();
                registerSize = 16;
                break;
            }
            #endif

            #if defined(__AVX__)
            if (supportedExtensions[2]) {
                myRegister.f256 = _mm256_setzero_ps();
                registerSize = 8;
                break;
            }
            #endif

            #if defined(__SSE__)
            if (supportedExtensions[0]) {
                myRegister.f128 = _mm_setzero_ps();
                registerSize = 4;
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Floats.");
            break;
    }
};

template <typename T>
vectorRegister<T>::~vectorRegister() {
    delete[] supportedExtensions;
};

template <typename T>
char vectorRegister<T>::getType() const {
    return type;
};

template <typename T>
int vectorRegister<T>::getRegisterSize() const {
    return registerSize;
};

template <typename T>
void vectorRegister<T>::zeroRegister() {
    loadedValues = 0;
    switch(type) {
        case 'i':
            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                myRegister.i512 = _mm512_setzero_si512();
                break;
            }
            #endif

            #if defined(__AVX2__)
            if (supportedExtensions[3]) {
                myRegister.i256 = _mm256_setzero_si256();
                break;
            }
            #endif

            #if defined(__SSE2__)
            if (supportedExtensions[1]) {
                myRegister.i128 = _mm_setzero_si128();
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Integers.");
            break;

        case 'd':
            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                myRegister.d512 = _mm512_setzero_pd();
                break;
            }
            #endif

            #if defined(__AVX__)
            if (supportedExtensions[2]) {
                myRegister.d256 = _mm256_setzero_pd();
                break;
            }
            #endif

            #if defined(__SSE2__)
            if (supportedExtensions[1]) {
                myRegister.d128 = _mm_setzero_pd();
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Doubles.");
            break;

        case 'f':
            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                myRegister.f512 = _mm512_setzero_ps();
                break;
            }
            #endif

            #if defined(__AVX__)
            if (supportedExtensions[2]) {
                myRegister.f256 = _mm256_setzero_ps();
                break;
            }
            #endif

            #if defined(__SSE__)
            if (supportedExtensions[0]) {
                myRegister.f128 = _mm_setzero_ps();
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Floats.");
            break;
    }
    return;
}

template <typename T>
void vectorRegister<T>::loadRegister(const vector<T> &itemsToLoad) {
    if (itemsToLoad.size() > registerSize) {
        throw invalid_argument("Size was too large of a vector to load into a register.");
    }

    switch(type) {
        case 'i': {
            alignas(64) int buffer[registerSize];
            for (int i = 0; i < itemsToLoad.size(); i++) {
                buffer[i] = itemsToLoad[i];
            }

            if (itemsToLoad.size() < registerSize) {
                // pad the vector to fit the full size.
                for (int i = itemsToLoad.size(); i < registerSize; i++) {
                    buffer[i] = 0;
                }
            }

            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                myRegister.i512 = _mm512_loadu_si512((__m512i*)buffer);
                break;
            }
            #endif

            #if defined(__AVX2__)
            if (supportedExtensions[3]) {
                myRegister.i256 = _mm256_loadu_si256((__m256i*)buffer);
                break;
            }
            #endif

            #if defined(__SSE2__)
            if (supportedExtensions[1]) {
                myRegister.i128 = _mm_loadu_si128((__m128i*)buffer);
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Integers.");
            break;
        }
        break;
        case 'd': {
            alignas(64) double buffer[registerSize];
            for (int i = 0; i < itemsToLoad.size(); i++) {
                buffer[i] = itemsToLoad[i];
            }

            if (itemsToLoad.size() < registerSize) {
                // pad the vector to fit the full size.
                for (int i = itemsToLoad.size(); i < registerSize; i++) {
                    buffer[i] = 0.0;
                }
            }
            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                myRegister.d512 = _mm512_loadu_pd(buffer);
                break;
            }
            #endif

            #if defined(__AVX__)
            if (supportedExtensions[2]) {
                myRegister.d256 = _mm256_loadu_pd(buffer);
                break;
            }
            #endif

            #if defined(__SSE2__)
            if (supportedExtensions[1]) {
                myRegister.d128 = _mm_loadu_pd(buffer);
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Doubles.");
            break;
        }
        break;
        case 'f': {
            alignas(64) float buffer[registerSize];
            for (int i = 0; i < itemsToLoad.size(); i++) {
                buffer[i] = itemsToLoad[i];
            }

            if (itemsToLoad.size() < registerSize) {
                // pad the vector to fit the full size.
                for (int i = itemsToLoad.size(); i < registerSize; i++) {
                    buffer[i] = 0;
                }
            }
            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                myRegister.f512 = _mm512_loadu_ps(buffer);
                break;
            }
            #endif

            #if defined(__AVX__)
            if (supportedExtensions[2]) {
                myRegister.f256 = _mm256_loadu_ps(buffer);
                break;
            }
            #endif

            #if defined(__SSE__)
            if (supportedExtensions[0]) {
                myRegister.f128 = _mm_loadu_ps(buffer);
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Floats.");
            break;
        }
        break;
    }

    loadedValues = itemsToLoad.size();
    
}

template <typename T>
vector<T> vectorRegister<T>::dumpRegister() const {
    switch(type) {
        case 'i': {
            alignas(64) int buffer[registerSize];

            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                _mm512_storeu_si512((__m512i*)buffer, myRegister.i512);
                vector<T> outputBuffer(loadedValues);
                for (int i = 0; i < loadedValues; i++) {
                outputBuffer[i] = buffer[i];
                } 
                return outputBuffer;
                break;
            }
            #endif

            #if defined(__AVX2__)
            if (supportedExtensions[3]) {
                _mm256_storeu_si256((__m256i*)buffer, myRegister.i256);
                vector<T> outputBuffer(loadedValues);
                for (int i = 0; i < loadedValues; i++) {
                outputBuffer[i] = buffer[i];
                } 
                return outputBuffer;
                break;
            }
            #endif

            #if defined(__SSE2__)
            if (supportedExtensions[1]) {
                _mm_storeu_si128((__m128i*)buffer, myRegister.i128);
                vector<T> outputBuffer(loadedValues);
                for (int i = 0; i < loadedValues; i++) {
                outputBuffer[i] = buffer[i];
                } 
                return outputBuffer;
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Integers.");
            break;
        }
        break;
        case 'd': {
            alignas(64) double buffer[registerSize];

            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                _mm512_storeu_pd(buffer, myRegister.d512);
                vector<T> outputBuffer(loadedValues);
                for (int i = 0; i < loadedValues; i++) {
                outputBuffer[i] = buffer[i];
                } 
                return outputBuffer;
                break;
            }
            #endif

            #if defined(__AVX__)
            if (supportedExtensions[2]) {
                _mm256_storeu_pd(buffer, myRegister.d256);
                vector<T> outputBuffer(loadedValues);
                for (int i = 0; i < loadedValues; i++) {
                outputBuffer[i] = buffer[i];
                } 
                return outputBuffer;
                break;
            }
            #endif

            #if defined(__SSE2__)
            if (supportedExtensions[1]) {
                _mm_storeu_pd(buffer, myRegister.d128);
                vector<T> outputBuffer(loadedValues);
                for (int i = 0; i < loadedValues; i++) {
                outputBuffer[i] = buffer[i];
                } 
                return outputBuffer;
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Doubles.");
            break;
        }
        break;
        case 'f': {
            alignas(64) float buffer[registerSize];

            #if defined(__AVX512F__)
            if (supportedExtensions[4]) {
                _mm512_storeu_ps(buffer, myRegister.f512);
                vector<T> outputBuffer(loadedValues);
                for (int i = 0; i < loadedValues; i++) {
                outputBuffer[i] = buffer[i];
                } 
                return outputBuffer;
                break;
            }
            #endif

            #if defined(__AVX__)
            if (supportedExtensions[2]) {
                _mm256_storeu_ps(buffer, myRegister.f256);
                vector<T> outputBuffer(loadedValues);
                for (int i = 0; i < loadedValues; i++) {
                outputBuffer[i] = buffer[i];
                } 
                return outputBuffer;
                break;
            }
            #endif

            #if defined(__SSE__)
            if (supportedExtensions[0]) {
                _mm_storeu_ps(buffer, myRegister.f128);
                vector<T> outputBuffer(loadedValues);
                for (int i = 0; i < loadedValues; i++) {
                outputBuffer[i] = buffer[i];
                } 
                return outputBuffer;
                break;
            }
            #endif

            throw invalid_argument("You do not support any SIMD for Floats.");
            break;
        }
        break;
    }
    vector<T> failedVector(0);
    return failedVector;
}

template<typename T>
void vectorRegister<T>::setLoadedValues(int newVals) {
    loadedValues = newVals;
    return;
}

template<typename T>
int vectorRegister<T>::getLoadedValues() const {
    return loadedValues;
}



template<typename T>
vectorRegister<T> vectorRegister<T>::operator+(const vectorRegister<T>& rhs) const {
    if (type != rhs.type) {
        // @TODO : add this functionality?
        throw invalid_argument("You cannot add two vector registers of different types.");
    }
    if (registerSize != rhs.registerSize) {
        // @TODO : add this functionality?
        throw invalid_argument("You cannot add two vector registers of different sizes.");
    }

    vectorRegister<T> solutionRegister;
    switch(type) {
            case 'i': {
                #if defined(__AVX512F__)
                if (supportedExtensions[4]) {
                    solutionRegister.myRegister.i512 = _mm512_add_epi32(myRegister.i512, rhs.myRegister.i512);
                    break;
                }
                #endif

                #if defined(__AVX2__)
                if (supportedExtensions[3]) {
                    solutionRegister.myRegister.i256 = _mm256_add_epi32(myRegister.i256, rhs.myRegister.i256);
                    break;
                }
                #endif

                #if defined(__SSE2__)
                if (supportedExtensions[1]) {
                    solutionRegister.myRegister.i128 = _mm_add_epi32(myRegister.i128, rhs.myRegister.i128);
                    break;
                }
                #endif

                throw invalid_argument("You do not support any SIMD for Integers.");
                break;
            }
            break;
            case 'd': {
                #if defined(__AVX512F__)
                if (supportedExtensions[4]) {
                    solutionRegister.myRegister.d512 = _mm512_add_pd(myRegister.d512, rhs.myRegister.d512);
                    break;
                }
                #endif

                #if defined(__AVX__)
                if (supportedExtensions[2]) {
                    solutionRegister.myRegister.d256 = _mm256_add_pd(myRegister.d256, rhs.myRegister.d256);
                    break;
                }
                #endif

                #if defined(__SSE2__)
                if (supportedExtensions[1]) {
                    solutionRegister.myRegister.d128 = _mm_add_pd(myRegister.d128, rhs.myRegister.d128);
                    break;
                }
                #endif

                throw invalid_argument("You do not support any SIMD for Doubles.");
                break;
            }
            break;
            case 'f': {
                #if defined(__AVX512F__)
                if (supportedExtensions[4]) {
                    solutionRegister.myRegister.f512 = _mm512_add_ps(myRegister.f512, rhs.myRegister.f512);
                    break;
                }
                #endif

                #if defined(__AVX__)
                if (supportedExtensions[2]) {
                    solutionRegister.myRegister.f256 = _mm256_add_ps(myRegister.f256, rhs.myRegister.f256);
                    break;
                }
                #endif

                #if defined(__SSE__)
                if (supportedExtensions[0]) {
                    solutionRegister.myRegister.f128 = _mm_add_ps(myRegister.f128, rhs.myRegister.f128);
                    break;
                }
                #endif

                throw invalid_argument("You do not support any SIMD for Floats.");
                break;


            }
            break;
        }
    if (loadedValues >= rhs.loadedValues) {
        solutionRegister.loadedValues = loadedValues;
    } else {
        solutionRegister.loadedValues = rhs.loadedValues;
    }
    return solutionRegister;
}

template class vectorRegister<int>;
template class vectorRegister<float>;
template class vectorRegister<double>;
