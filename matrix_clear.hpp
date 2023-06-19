#ifndef matrix_clear
#define matrix_clear

#include <iostream>    // для использования std::cout, std::cin
#include <thread>      // для использования std::thread
#include <algorithm>   // для использования std::min
#include <vector>      // для использования std::vector
#include <future>      // для использования std::future и std::async
#include <stdexcept>   // для использования std::invalid_argument
#include <fstream>     // для использования std::ifstream и std::ofstream
#include <sstream>     // для использования std::stringstream
#include <mutex>       // для использования std::mutex и std::unique_lock
template <typename T>
class Matrix {
private:
    int rows_;
    int cols_;
    T* data_;
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
        data_ = new T[rows * cols];
    }

    Matrix(const std::string& filename) {
        std::ifstream file(filename);
        if (file.is_open()) {
            file >> rows_ >> cols_;
            data_ = new T[rows_ * cols_];
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < cols_; j++) {
                    file >> data_[i * cols_ + j];
                }
            }
            file.close();
        }
    }

    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(new T[rows_ * cols_]) {
    std::copy(other.data_, other.data_ + rows_ * cols_, data_);
    }

    Matrix(int rows, int cols, T scalar) : rows_(rows), cols_(cols), data_(new T[rows * cols]) {
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            data_[i * cols_ + j] = scalar;
            }
        }
    }

    ~Matrix() {
        delete[] data_;
    }


    void print() const {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                std::cout << data_[i * cols_ + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    void read_from_console() {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                std::cin >> data_[i * cols_ + j];
            }
        }
    }
    void read_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (file.is_open()) {
            file >> rows_;
            file >> cols_;
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < cols_; j++) {
                    file >> data_[i * cols_ + j];
                }
            }
            file.close();
        }
    }
    void write_to_file(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << rows_ << std::endl;
            file << cols_ << std::endl;
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < cols_; j++) {
                    file << data_[i * cols_ + j] << " ";
                }
                file << std::endl;
            }
            file.close();
        }
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
        delete[] data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = new T[rows_ * cols_];
        std::copy(other.data_, other.data_ + rows_ * cols_, data_);
        }
        return *this;
    }

    Matrix<T> operator*(const Matrix<T>& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        Matrix<T> result(rows_, other.cols_);
        for (int i = 0; i < result.rows_; i++) {
            for (int j = 0; j < result.cols_; j++) {
                T sum = 0;
                for (int k = 0; k < cols_; k++) {
                    sum += data_[i * cols_ + k] * other.data_[k * cols_ + j];
                }
                result.data_[i * cols_ + j] = sum;
            }
        }
        return result;
    }


    Matrix<T> mult(const Matrix<T>& other, unsigned int num_threads) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix<T> result(rows_, other.cols_);

    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.push_back(std::thread([&, i]() {
            for (int r = i; r < rows_; r += num_threads) {
                for (int c = 0; c < other.cols_; ++c) {
                    T sum = 0;
                    for (int k = 0; k < cols_; ++k) {
                        sum += data_[r * cols_ + k] * other.data_[k * other.cols_ + c];
                    }
                    result.data_[r * other.cols_ + c] = sum;
                }
            }
        }));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}


   std::future<Matrix<T>> mult(const Matrix<T>& other, unsigned int num_threads, unsigned int block_size) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    auto result = std::make_shared<Matrix<T>>(rows_, other.cols_);

    std::vector<std::future<void>> futures;

    int block_size_int = static_cast<int>(block_size); // Приведение block_size к типу int

    for (unsigned int t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&, t]() {
            for (int i_block = 0; i_block < rows_; i_block += block_size_int) {
                for (int j_block = 0; j_block < other.cols_; j_block += block_size_int) {
                    for (int r = i_block; r < std::min(i_block + block_size_int, rows_); ++r) {
                        for (int c = j_block; c < std::min(j_block + block_size_int, other.cols_); ++c) {
                            T sum = 0;
                            for (int k = t; k < cols_; k += num_threads) {
                                sum += data_[r * cols_ + k] * other.data_[k * other.cols_ + c];
                            }
                            (*result).data_[r * other.cols_ + c] = sum;
                        }
                    }
                }
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    return std::async(std::launch::deferred, [result]() { return *result; });
}



    Matrix<T> operator*(const T& scalar) const {
        Matrix<T> result(*this);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                result.data_[i * cols_ + j] *= scalar;
            }
        }
        return result;
    }

    Matrix<T> mult_scalar(T scalar, unsigned int num_threads) const {
    Matrix<T> result(rows_, cols_);

    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.push_back(std::thread([&, i]() {
            for (int r = i; r < rows_ * cols_; r += num_threads) {
                result.data_[r] = this->data_[r] * scalar;
            }
        }));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
    }

    std::future<Matrix<T>> mult_scalar(T scalar, unsigned int num_threads, unsigned int block_size) const {
    auto result = std::make_shared<Matrix<T>>(rows_, cols_);

    std::vector<std::future<void>> futures;

    int block_size_int = static_cast<int>(block_size); // Приведение block_size к типу int

    for (unsigned int t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&, t]() {
            for (int i_block = 0; i_block < rows_; i_block += block_size_int) {
                for (int j_block = 0; j_block < cols_; j_block += block_size_int) {
                    for (int r = i_block; r < std::min(i_block + block_size_int, rows_); ++r) {
                        for (int c = j_block; c < std::min(j_block + block_size_int, cols_); ++c) {
                            (*result).data_[r * cols_ + c] = this->data_[r * cols_ + c] * scalar;
                        }
                    }
                }
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    return std::async(std::launch::deferred, [result]() { return *result; });
}

    Matrix<T> operator+(const Matrix<T>& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
        Matrix<T> result(rows_, cols_);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                result.data_[i * cols_ + j] = data_[i * cols_ + j] + other.data_[i * cols_ + j];
            }
        }
        return result;
    }

    Matrix<T> thread_sum(const Matrix<T>& other, unsigned int num_threads) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }
    Matrix<T> result(rows_, cols_);
    std::vector<std::thread> threads(num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        threads[t] = std::thread([this, &other, &result, t, num_threads]() {
            for (int j = t; j < cols_; j += num_threads) {
                for (int i = 0; i < this->rows_; i++) {
                    result.data_[i * cols_ + j] = this->data_[i * cols_ + j] + other.data_[i * cols_ + j];
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
    }

    std::future<Matrix<T>> async_thread_sum(const Matrix<T>& other, unsigned int num_threads, unsigned int block_size) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }

    auto result = std::make_shared<Matrix<T>>(rows_, cols_);

    std::vector<std::future<void>> futures;

    int block_size_int = static_cast<int>(block_size); // Приведение block_size к типу int

    for (unsigned int t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&, t]() {
            for (int i_block = 0; i_block < rows_; i_block += block_size_int) {
                for (int j_block = 0; j_block < cols_; j_block += block_size_int) {
                    for (int r = i_block; r < std::min(i_block + block_size_int, rows_); ++r) {
                        for (int c = j_block; c < std::min(j_block + block_size_int, cols_); ++c) {
                            (*result).data_[r * cols_ + c] = this->data_[r * cols_ + c] + other.data_[r * cols_ + c];
                        }
                    }
                }
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    return std::async(std::launch::deferred, [result]() { return *result; });
}


    Matrix<T> operator-(const Matrix<T>& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }
        Matrix<T> result(rows_, cols_);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                result.data_[i * cols_ + j] = data_[i * cols_ + j] - other.data_[i * cols_ + j];
            }
        }
        return result;
    }

     Matrix<T> thread_diff(const Matrix<T>& other, unsigned int num_threads) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }
    Matrix<T> result(rows_, cols_);
    std::vector<std::thread> threads(num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        threads[t] = std::thread([this, &other, &result, t, num_threads]() {
            for (int j = t; j < cols_; j += num_threads) {
                for (int i = 0; i < this->rows_; i++) {
                    result.data_[i * cols_ + j] = this->data_[i * cols_ + j] + other.data_[i * cols_ + j];
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
    }

    std::future<Matrix<T>> async_thread_diff(const Matrix<T>& other, unsigned int num_threads, unsigned int block_size) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }

    auto result = std::make_shared<Matrix<T>>(rows_, cols_);

    std::vector<std::future<void>> futures;

    int block_size_int = static_cast<int>(block_size); // Приведение block_size к типу int

    for (unsigned int t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&, t]() {
            for (int i_block = 0; i_block < rows_; i_block += block_size_int) {
                for (int j_block = 0; j_block < cols_; j_block += block_size_int) {
                    for (int r = i_block; r < std::min(i_block + block_size_int, rows_); ++r) {
                        for (int c = j_block; c < std::min(j_block + block_size_int, cols_); ++c) {
                            (*result).data_[r * cols_ + c] = this->data_[r * cols_ + c] - other.data_[r * cols_ + c];
                        }
                    }
                }
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    return std::async(std::launch::deferred, [result]() { return *result; });
}

   Matrix<T> operator!() const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Matrix must be square to have an inverse");
        }
        Matrix<T> result(rows_, cols_);
        Matrix<T> tmp(*this);
        for (int i = 0; i < rows_; i++) {
            result.data_[i * cols_ + i] = 1;
        }
        for (int i = 0; i < rows_; i++) {
            if (tmp.data_[i * cols_ + i] == 0) {
                bool found_nonzero = false;
                for (int j = i + 1; j < rows_; j++) {
                    if (tmp.data_[j * cols_ + i] != 0) {
                        std::swap_ranges(tmp.data_ + i * cols_, tmp.data_ + (i+1) * cols_, tmp.data_ + j * cols_);
                        std::swap_ranges(result.data_ + i * cols_, result.data_ + (i+1) * cols_, result.data_ + j * cols_);
                        found_nonzero = true;
                        break;
                    }
                }
                if (!found_nonzero) {
                    throw std::invalid_argument("Matrix is not invertible");
                }
            }
            T pivot = tmp.data_[i * cols_ + i];
            for (int j = 0; j < cols_; j++) {
                tmp.data_[i * cols_ + j] /= pivot;
                result.data_[i * cols_ + j] /= pivot;
            }
            for (int j = 0; j < rows_; j++) {
                if (i != j) {
                    T multiplier = tmp.data_[j * cols_ + i];
                    for (int k = 0; k < cols_; k++) {
                        tmp.data_[j * cols_ + k] -= multiplier * tmp.data_[i * cols_ + k];
                        result.data_[j * cols_ + k] -= multiplier * result.data_[i * cols_ + k];
                    }
                }
            }
        }
        return result;
    }


    bool operator==(const Matrix<T>& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if (data_[i * cols_ + j] != other.data_[i * cols_ + j]) {
                return false;
            }
        }
    }
    return true;
    }

    bool operator!=(const Matrix<T>& other) const {
        return !(*this == other);
    }

    bool operator==(const T& scalar) const {
        if ((rows_ != 1 || cols_ != 1) && scalar != static_cast<T>(0) && scalar != static_cast<T>(1)) {
            return false;
        }
        if (scalar == static_cast<T>(0)) {
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < cols_; j++) {
                    if (data_[i * cols_ + j] != static_cast<T>(0)) {
                        return false;
                    }
                }
            }
        } else if (scalar == static_cast<T>(1)) {
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < cols_; j++) {
                    if (i == j && data_[i * cols_ + j] != static_cast<T>(1)) {
                        return false;
                    } else if (i != j && data_[i * cols_ + j] != static_cast<T>(0)) {
                        return false;
                    }
                }
            }
        } else {
            return *this == Matrix<T>(rows_, cols_, scalar);
        }
        return true;
    }

    bool operator!=(const T& scalar) const {
        return !(*this == scalar);
    }

    static Matrix<T> zeros(int rows, int cols) {
        Matrix<T> mat(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat(i, j) = static_cast<T>(0);
            }
        }
        return mat;
    }

    
    static Matrix<T> ones(int rows, int cols) {
        Matrix<T> mat(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat(i, j) = (i == j) ? static_cast<T>(1) : static_cast<T>(0);
            }
        }
        return mat;
    }

     T& operator()(const int row, const int col) {
        if (row > rows_ || col > cols_ || row < 0 || col < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data_[row * cols_ + col];
    }

    
    const T& operator()(const int row, const int col) const {
        if (row > rows_ || col > cols_ || row < 0 || col < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data_[row * cols_ + col];
    }
    
    
};

#endif