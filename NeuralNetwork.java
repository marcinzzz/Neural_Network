import java.util.Arrays;
import java.util.Random;

class NeuralNetwork {
    private Matrix[] weights;
    private Matrix[] biases;

    private float learningRate;

    private float trainingSets[][];
    private float targets[][];
    private int trainingLength;

    private int length;

    NeuralNetwork(int inputNodes, int hiddenLayersNodes[], int outputNodes, float learningRate) {
        this.learningRate = learningRate;
        this.length = hiddenLayersNodes.length;

        this.weights = new Matrix[length + 1];
        this.biases = new Matrix[length + 1];

        weights[0] = new Matrix(hiddenLayersNodes[0], inputNodes);
        weights[length] = new Matrix(outputNodes, hiddenLayersNodes[length - 1]);
        weights[0].randomize();
        weights[length].randomize();

        biases[0] = new Matrix(hiddenLayersNodes[0], 1);
        biases[length] = new Matrix(outputNodes, 1);
        biases[0].randomize();
        biases[length].randomize();

        for (int i = 1; i < length; i++) {
            weights[i] = new Matrix(hiddenLayersNodes[i], hiddenLayersNodes[i - 1]);
            biases[i] = new Matrix(hiddenLayersNodes[i], 1);

            weights[i].randomize();
            biases[i].randomize();
        }
    }

    NeuralNetwork(float trainingSets[][], float targets[][], int hiddenLayersNodes[], float learningRate, int trainingLength) {
        this(trainingSets[0].length, hiddenLayersNodes, targets[0].length, learningRate);

        this.trainingSets = trainingSets;
        this.targets = targets;
        this.trainingLength = trainingLength;
    }

    float[] feedForward(float input[]) {
        Matrix inputs = Matrix.fromArray(input);
        Matrix hiddenLayers[] = new Matrix[weights.length - 1];

        hiddenLayers[0] = Matrix.multiply(weights[0], inputs);
        hiddenLayers[0].add(biases[0]);
        hiddenLayers[0].map(Matrix.SIGMOID);

        for (int i = 1; i < length; i++) {
            hiddenLayers[i] = Matrix.multiply(weights[i], hiddenLayers[i - 1]);
            hiddenLayers[i].add(biases[i]);
            hiddenLayers[i].map(Matrix.SIGMOID);
        }

        Matrix output = Matrix.multiply(weights[length], hiddenLayers[length - 1]);
        output.add(biases[length]);
        output.map(Matrix.SIGMOID);

        return output.toArray();
    }

    void printResults(float input[]) {
        System.out.println(Arrays.toString(input) + " : " + Arrays.toString(feedForward(input)));
    }

    void train(float input[], float target[]) {
        Matrix inputs = Matrix.fromArray(input);
        Matrix hiddenLayers[] = new Matrix[length];

        hiddenLayers[0] = Matrix.multiply(weights[0], inputs);
        hiddenLayers[0].add(biases[0]);
        hiddenLayers[0].map(Matrix.SIGMOID);

        for (int i = 1; i < length; i++) {
            hiddenLayers[i] = Matrix.multiply(weights[i], hiddenLayers[i - 1]);
            hiddenLayers[i].add(biases[i]);
            hiddenLayers[i].map(Matrix.SIGMOID);
        }

        Matrix outputs = Matrix.multiply(weights[length], hiddenLayers[length - 1]);
        outputs.add(biases[length]);
        outputs.map(Matrix.SIGMOID);

        Matrix targets = Matrix.fromArray(target);

        Matrix outputErrors = Matrix.subtract(targets, outputs);

        Matrix gradients = Matrix.map(outputs, Matrix.DSIGMOID);
        gradients.multiply(outputErrors);
        gradients.multiply(learningRate);

        Matrix hiddenLayerTransposed = Matrix.transpose(hiddenLayers[length - 1]);
        Matrix weightsDeltas = Matrix.multiply(gradients, hiddenLayerTransposed);

        this.weights[length].add(weightsDeltas);
        this.biases[length].add(gradients);

        Matrix previousLayerErrors = outputErrors;

        for (int i = length - 1; i >= 0; i--) {
            Matrix weightsTransposed = Matrix.transpose(weights[i + 1]);
            Matrix hiddenErrors = Matrix.multiply(weightsTransposed, previousLayerErrors);

            Matrix hiddenGradient = Matrix.map(hiddenLayers[i], Matrix.DSIGMOID);
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

    void train() {
        if (trainingSets != null) {
            Random random = new Random();
            for (int i = 0; i < trainingLength; i++) {
                int r = random.nextInt(trainingSets.length);
                train(trainingSets[r], targets[r]);
            }
        } else {
            System.out.println("ERROR! CANNOT TRAIN NEURAL NETWORK, NO TRAINING DATA");
        }
    }
}
