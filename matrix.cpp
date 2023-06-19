#include "matrix_clear.hpp"

// int main() {
//     // Создание двух матриц размером 3x3
//     Matrix<int> matrix1(3, 3);
//     Matrix<int> matrix2(3, 3);

//     // Заполнение матриц значениями
//     for (int i = 0; i < 3; ++i) {
//         for (int j = 0; j < 3; ++j) {
//             matrix1(i, j) = i * 3 + j + 1;  // Заполнение матрицы значениями от 1 до 9
//             matrix2(i, j) = (i * 3 + j + 1) * 10;  // Заполнение матрицы значениями от 10 до 90
//         }
//     }

//     std::cout << "Matrix 1:" << std::endl;
//     matrix1.print();

//     std::cout << "\nMatrix 2:" << std::endl;
//     matrix2.print();

//     // Выполнение операций над матрицами
//     Matrix<int> sumMatrix = matrix1 + matrix2;
//     Matrix<int> differenceMatrix = matrix1 - matrix2;
//     Matrix<int> productMatrix = matrix1 * matrix2;
//     Matrix<int> scaledMatrix = matrix1 * 2;

//     // Вывод результатов
//     std::cout << "\nSum of matrices:" << std::endl;
//     sumMatrix.print();

//     std::cout << "\nDifference of matrices:" << std::endl;
//     differenceMatrix.print();

//     std::cout << "\nProduct of matrices:" << std::endl;
//     productMatrix.print();

//     std::cout << "\nMatrix 1 scaled by 2:" << std::endl;
//     scaledMatrix.print();

//     return 0;
// }



// int main() {
//     Matrix<double> a(2, 2);
//     Matrix<double> b(2, 2);
//     a(0,0) = 1;
//     a(0,1) = 0;
//     a(1,0) = 0;
//     a(1,1) = 1;
//     a.print();
//     std::cout << std::endl;
//     b(0,0) = 1;
//     b(0,1) = 2;
//     b(1,0) = 3;
//     b(1,1) = 4;
//     b.print();
//     std::cout << std::endl;
//     Matrix<double> m = a*b;
//     m.print();
//     std::cout<< std::endl;
//     Matrix<double> d = m-b;
//     d.print();


//     // int num_threads = std::thread::hardware_concurrency();
//     // std::cout<< num_threads;
    
//     // m.read_from_console();
//     // m(1,1) = 10;
//     // m.print();

//     // std::cout << std::endl;
//     // Matrix<double> A_inv = !m;
//     // A_inv.print();
//     // std::cout << std::endl;
//     // Matrix<double> M = m * A_inv;
//     // M.print();

//     // m.write_to_file("matrix.txt");

//     // std::cout << std::endl;
//     // Matrix<double> n("matrix.txt");
    
//     // n.print();
//     // std::cout << std::endl;
//     // Matrix<double> q(3, 3);
//     // q.read_from_file("matrix.txt");
//     // q.print();

//     // std::cout << std::endl;
//     // Matrix<int> myMatrix = Matrix<int>::zeros(3, 3);

//     // if (myMatrix == 0) {
//     // std::cout << "Matrix is zero!" << std::endl;
//     // } else {
//     // std::cout << "Matrix is not zero!" << std::endl;
//     // }
    
//     // Matrix<int> myMatrix1 = Matrix<int>::ones(2, 2);

//     // if (myMatrix1 == 1) {
//     // std::cout << "Matrix is identity!" << std::endl;
//     // } else {
//     // std::cout << "Matrix is not identity!" << std::endl;
//     // }

//     // if (myMatrix1 == myMatrix) {
//     // std::cout << "Matrix are equal" << std::endl;
//     // } else {
//     // std::cout << "Matrix are not equal!" << std::endl;
//     // }

//     return 0;
// }



// #include <iostream>
// #include <chrono>
// #include <random>

// int main() {
//     std::default_random_engine generator;
//     std::uniform_real_distribution<double> distribution(0.0,1.0);

//     for (int size = 100; size <= 1000; size+=100) {
//         Matrix<double> mat1(size, size), mat2(size, size);

//         for (int i = 0; i < size; i++) {
//             for (int j = 0; j < size; j++) {
//                 mat1(i, j) = distribution(generator);
//                 mat2(i, j) = distribution(generator);
//             }
//         }
    
//         // auto start = std::chrono::high_resolution_clock::now();
//         // Matrix<double> result1 = mat1 * mat2; // операция умножения в одном потоке
//         // auto end = std::chrono::high_resolution_clock::now();
//         // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//         // std::cout << "Time taken by single-threaded multiplication of " << size << "x" << size << " matrices: " << duration << " microseconds" << std::endl;

//         // start = std::chrono::high_resolution_clock::now();
//         // Matrix<double> result2 = mat1.mult(mat2, 16); // операция умножения в 16 потоках
//         // end = std::chrono::high_resolution_clock::now();
//         // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//         // std::cout << "Time taken by multi-threaded multiplication of " << size << "x" << size << " matrices: " << duration << " microseconds" << std::endl;

//     // }

//         // Используем 8 потоков и размер блока 100
//     unsigned int num_threads = 8;
//     unsigned int block_size = 100;

//     // Измеряем время выполнения умножения
//     auto start = std::chrono::high_resolution_clock::now();
//     std::future<Matrix<double>> result_future = mat1.async_thread_sum(mat2, num_threads, block_size);
//     Matrix<double> result = result_future.get();
//     auto stop = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = stop - start;
    
//     std::cout << "Time taken by block-wise multi-threaded multiplication: " << elapsed.count() << " seconds" << std::endl;
    
//     }
//     return 0;
// }




#include <chrono>
#include <iostream>
#include "matrix_clear.hpp"  // предположим, что ваш класс Matrix определен в этом файле

double AmdahlsLaw(double P, double n) {
    double S = 1.0 - P;
    return 1.0 / (S + P/n);
}

int main() {
    // Создаем матрицы и скаляр
    Matrix<int> A(1000, 1000, 1);  // создаем матрицу размера 1000x1000, заполненную 1
    Matrix<int> B(1000, 1000, 1);  // создаем матрицу размера 1000x1000, заполненную 1
    int scalar = 2;

    double P = 1.0;  // предположим, что вся работа может быть выполнена параллельно

    // Вектор с количеством потоков, которые мы хотим проверить
    std::vector<int> num_threads = {1, 2, 4, 8, 16};

    // Замеряем время выполнения функций для различного количества потоков
    for (auto& n : num_threads) {
        // сложение матриц
        auto start = std::chrono::high_resolution_clock::now();
        auto result = A.thread_sum(B, n);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double theoretical = AmdahlsLaw(P, n);
        std::cout << "Addition with " << n << " threads took " << duration.count() << " ms. Theoretical speedup: " << theoretical << "\n";

        // вычитание матриц
        start = std::chrono::high_resolution_clock::now();
        result = A.thread_diff(B, n);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        theoretical = AmdahlsLaw(P, n);
        std::cout << "Subtraction with " << n << " threads took " << duration.count() << " ms. Theoretical speedup: " << theoretical << "\n";

        // умножение матриц
        start = std::chrono::high_resolution_clock::now();
        result = A.mult(B, n);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        theoretical = AmdahlsLaw(P, n);
        std::cout << "Multiplication with " << n << " threads took " << duration.count() << " ms. Theoretical speedup: " << theoretical << "\n";

        // умножение матрицы на скаляр
        start = std::chrono::high_resolution_clock::now();
        result = A.mult_scalar(scalar, n);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        theoretical = AmdahlsLaw(P, n);
        std::cout << "Scalar multiplication with " << n << " threads took " << duration.count() << " ms. Theoretical speedup: " << theoretical << "\n";
    }

    return 0;
}


