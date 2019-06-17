import java.util.Arrays;
import java.util.Random;

class Matrix {
    private int rows;
    private int cols;
    private float data[][];

    static Map SIGMOID = v -> 1 / (float)(1 + Math.exp(-v));
    static Map DSIGMOID = v -> v * (1 - v);

    Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows][cols];

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] = 0;
            }
        }
    }

    static Matrix fromArray(float[] arr) {
        Matrix m = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }

    static Matrix subtract(Matrix a, Matrix b) {
        // Return a new Matrix a-b
        Matrix result = new Matrix(a.rows, a.cols);
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    float[] toArray() {
        float arr[];
        if (rows >= cols) {
            arr = new float[rows];
            for (int i = 0; i < this.rows; i++) {
                arr[i] = this.data[i][0];
            }
        } else {
            arr = new float[cols];
            for (int j = 0; j < this.cols; j++) {
                arr[j] = this.data[0][j];
            }
        }
        return arr;
    }

    void randomize() {
        Random random = new Random();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] = random.nextFloat();
                if (random.nextBoolean())
                    this.data[i][j] *= -1;
            }
        }
    }

    void add(Matrix n) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] += n.data[i][j];
            }
        }
    }

    void add(float n) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] += n;
            }
        }
    }

    static Matrix transpose(Matrix matrix) {
        Matrix result = new Matrix(matrix.cols, matrix.rows);
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                result.data[j][i] = matrix.data[i][j];
            }
        }
        return result;
    }

    static Matrix multiply(Matrix a, Matrix b) {
        // Matrix product
        if (a.cols != b.rows) {
            System.out.println("ERROR");
            return null;
        }
        Matrix result = new Matrix(a.rows, b.cols);
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                // Dot product of values in col
                float sum = 0;
                for (int k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    void multiply(Matrix n) {
        // hadamard product
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] *= n.data[i][j];
            }
        }
    }

    void multiply(float n) {
        // Scalar product
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] *= n;
            }
        }
    }

    void map(Map map) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] = map.map(this.data[i][j]);
            }
        }
    }

    static Matrix map(Matrix matrix, Map map) {
        Matrix result = new Matrix(matrix.rows, matrix.cols);
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                float val = matrix.data[i][j];
                result.data[i][j] = map.map(val);
            }
        }
        return result;
    }

    void print() {
        for (int i = 0; i < this.rows; i++) {
            System.out.println(Arrays.toString(this.data[i]));
        }
        System.out.println();
    }
}

