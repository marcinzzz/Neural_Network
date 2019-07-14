import java.io.File;
import java.util.Arrays;

class NeuralNetwork {
    private Matrix[] weights;
    private Matrix[] biases;
    private int layers;
    private float learningRate;

    static Map SIGMOID = v -> 1 / (float)(1 + Math.exp(-v));
    static Map DSIGMOID = v -> v * (1 - v);

    NeuralNetwork(int inputNodes, int hiddenLayersNodes[], int outputNodes, float learningRate) {
        this.learningRate = learningRate;
        this.layers = hiddenLayersNodes.length;

        this.weights = new Matrix[layers + 1];
        this.biases = new Matrix[layers + 1];

        weights[0] = new Matrix(hiddenLayersNodes[0], inputNodes);
        weights[layers] = new Matrix(outputNodes, hiddenLayersNodes[layers - 1]);
        weights[0].randomize();
        weights[layers].randomize();

        biases[0] = new Matrix(hiddenLayersNodes[0], 1);
        biases[layers] = new Matrix(outputNodes, 1);
        biases[0].randomize();
        biases[layers].randomize();

        for (int i = 1; i < layers; i++) {
            weights[i] = new Matrix(hiddenLayersNodes[i], hiddenLayersNodes[i - 1]);
            biases[i] = new Matrix(hiddenLayersNodes[i], 1);

            weights[i].randomize();
            biases[i].randomize();
        }
    }

    NeuralNetwork(int inputNodes, int hiddenLayersNodes[], int outputNodes, Map activationFunctions[], float learningRate) {
        if (activationFunctions.length != hiddenLayersNodes.length + 1) {
            System.out.println("ERROR, cannot create neural network");
            return;
        }

        this.learningRate = learningRate;
        this.layers = hiddenLayersNodes.length;

        this.weights = new Matrix[layers + 1];
        this.biases = new Matrix[layers + 1];

        weights[0] = new Matrix(hiddenLayersNodes[0], inputNodes);
        weights[layers] = new Matrix(outputNodes, hiddenLayersNodes[layers - 1]);
        weights[0].randomize();
        weights[layers].randomize();

        biases[0] = new Matrix(hiddenLayersNodes[0], 1);
        biases[layers] = new Matrix(outputNodes, 1);
        biases[0].randomize();
        biases[layers].randomize();

        for (int i = 1; i < layers; i++) {
            weights[i] = new Matrix(hiddenLayersNodes[i], hiddenLayersNodes[i - 1]);
            biases[i] = new Matrix(hiddenLayersNodes[i], 1);

            weights[i].randomize();
            biases[i].randomize();
        }
    }

    NeuralNetwork(String directory, float learningRate) {
        this.learningRate = learningRate;

        File dir = new File(directory);
        File list[] = dir.listFiles();

        if (list != null) {
            int length = list.length;

            this.weights = new Matrix[length / 2];
            this.biases = new Matrix[length / 2];
            this.layers = list.length / 2 - 1;

            for (int i = 0; i < length / 2; i++) {
                String fileName = dir + "\\" + list[i].getName();
                biases[i] = new Matrix(fileName);
            }

            for (int i = length / 2; i < length; i++) {
                String fileName = dir + "\\" + list[i].getName();
                weights[i - length / 2] = new Matrix(fileName);
            }
        }
    }

    float[] feedForward(float input[]) {
        Matrix inputs = Matrix.fromArray(input);
        Matrix hiddenLayers[] = new Matrix[weights.length - 1];

        hiddenLayers[0] = Matrix.multiply(weights[0], inputs);
        hiddenLayers[0].add(biases[0]);
        hiddenLayers[0].map(SIGMOID);

        for (int i = 1; i < layers; i++) {
            hiddenLayers[i] = Matrix.multiply(weights[i], hiddenLayers[i - 1]);
            hiddenLayers[i].add(biases[i]);
            hiddenLayers[i].map(SIGMOID);
        }

        Matrix output = Matrix.multiply(weights[layers], hiddenLayers[layers - 1]);
        output.add(biases[layers]);
        output.map(SIGMOID);

        return output.toArray();
    }

    void printResults(float input[]) {
        System.out.println(Arrays.toString(input) + " : " + Arrays.toString(feedForward(input)));
    }

    void train(float input[], float target[]) {
        Matrix inputs = Matrix.fromArray(input);
        Matrix hiddenLayers[] = new Matrix[layers];

        hiddenLayers[0] = Matrix.multiply(weights[0], inputs);
        hiddenLayers[0].add(biases[0]);
        hiddenLayers[0].map(SIGMOID);

        for (int i = 1; i < layers; i++) {
            hiddenLayers[i] = Matrix.multiply(weights[i], hiddenLayers[i - 1]);
            hiddenLayers[i].add(biases[i]);
            hiddenLayers[i].map(SIGMOID);
        }

        Matrix outputs = Matrix.multiply(weights[layers], hiddenLayers[layers - 1]);
        outputs.add(biases[layers]);
        outputs.map(SIGMOID);

        Matrix targets = Matrix.fromArray(target);

        Matrix outputErrors = Matrix.subtract(targets, outputs);

        Matrix gradients = Matrix.map(outputs, DSIGMOID);
        gradients.multiply(outputErrors);
        gradients.multiply(learningRate);

        Matrix hiddenLayerTransposed = Matrix.transpose(hiddenLayers[layers - 1]);
        Matrix weightsDeltas = Matrix.multiply(gradients, hiddenLayerTransposed);

        this.weights[layers].add(weightsDeltas);
        this.biases[layers].add(gradients);

        Matrix previousLayerErrors = outputErrors;

        for (int i = layers - 1; i >= 0; i--) {
            Matrix weightsTransposed = Matrix.transpose(weights[i + 1]);
            Matrix hiddenErrors = Matrix.multiply(weightsTransposed, previousLayerErrors);

            Matrix hiddenGradient = Matrix.map(hiddenLayers[i], DSIGMOID);
            hiddenGradient.multiply(hiddenErrors);
            hiddenGradient.multiply(learningRate);

            Matrix inputsTransposed;
            if (i != 0)
                inputsTransposed = Matrix.transpose(hiddenLayers[i - 1]);
            else
                inputsTransposed = Matrix.transpose(inputs);
            Matrix weightsDeltas2 = Matrix.multiply(hiddenGradient, inputsTransposed);

            this.weights[i].add(weightsDeltas2);
            this.biases[i].add(hiddenGradient);

            previousLayerErrors = hiddenErrors;
        }
    }

    void export(String directory) {
        new File(directory).mkdir();
        for (int i = 0; i < weights.length; i++) {
            weights[i].exportToFile(directory + "\\weights" + i + ".txt");
        }
        for (int i = 0; i < biases.length; i++) {
            biases[i].exportToFile(directory + "\\biases" + i + ".txt");
        }
    }
}