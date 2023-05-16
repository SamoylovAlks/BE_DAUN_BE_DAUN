#include <iostream>
#include <fstream>

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
    void set(int i, int j, T value) {
        data_[i * cols_ + j] = value;
    }
    T get(int i, int j) const {
        return data_[i * cols_ + j];
    }
    int rows() const {
        return rows_;
    }
    int cols() const {
        return cols_;
    }
    void print() const {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                std::cout << get(i, j) << " ";
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
                    file << get(i, j) << " ";
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

    Matrix<T> operator*(const T& scalar) const {
        Matrix<T> result(*this);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                result.data_[i * cols_ + j] *= scalar;
            }
        }
        return result;
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



    // Перегрузка оператора равенства для проверки на равенство матриц
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
        if (row >= rows_ || col >= cols_ || row < 0 || col < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data_[row * cols_ + col];
    }

    
    const T& operator()(const int row, const int col) const {
        if (row >= rows_ || col >= cols_ || row < 0 || col < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data_[row * cols_ + col];
    }
};
int main() {
    Matrix<double> m(2, 2);
    m.read_from_console();
    m(1,1) = 10;
    m.print();

    std::cout << std::endl;
    Matrix<double> A_inv = !m;
    A_inv.print();
    std::cout << std::endl;
    Matrix<double> M = m * A_inv;
    M.print();

    m.write_to_file("matrix.txt");

    std::cout << std::endl;
    Matrix<double> n("matrix.txt");
    
    n.print();
    std::cout << std::endl;
    Matrix<double> q(3, 3);
    q.read_from_file("matrix.txt");
    q.print();

    std::cout << std::endl;
    Matrix<int> myMatrix = Matrix<int>::zeros(3, 3);

    if (myMatrix == 0) {
    std::cout << "Matrix is zero!" << std::endl;
    } else {
    std::cout << "Matrix is not zero!" << std::endl;
    }
    
    Matrix<int> myMatrix1 = Matrix<int>::ones(2, 2);

    if (myMatrix1 == 1) {
    std::cout << "Matrix is identity!" << std::endl;
    } else {
    std::cout << "Matrix is not identity!" << std::endl;
    }

    if (myMatrix1 == myMatrix) {
    std::cout << "Matrix are equal" << std::endl;
    } else {
    std::cout << "Matrix are not equal!" << std::endl;
    }

    return 0;
}
