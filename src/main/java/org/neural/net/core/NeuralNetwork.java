package org.neural.net.core;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.jblas.DoubleMatrix;
import org.neural.net.functions.ActivationFunction;
import org.neural.net.functions.Sigmoid;
import org.neural.net.layer.Layer;
import org.neural.net.utils.MathUtils;

import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("all")
public class NeuralNetwork implements CoreNet {

    private final List<Layer> layers;

    private static MathUtils mathUtils = new MathUtils();

    private List<RealMatrix> weightingFactors;
    private ActivationFunction activationFunction = new Sigmoid();
    private boolean logger = false;
    private double learningRate = 0.1;

    public NeuralNetwork(List<Layer> layers, List<RealMatrix> weightingFactors) {

        if (layers == null || layers.isEmpty() || layers.size() == 1) {
            throw new RuntimeException("Incorrect layers size. Must be at least 2");
        }
        this.layers = layers;
        this.weightingFactors = weightingFactors;

        if (weightingFactors.isEmpty()) {
            this.generateFactors();
        }
    }

    public NeuralNetwork(List<Layer> layers) {
        this(layers, new ArrayList<>());
    }

    @Override
    public void train(RealMatrix input, RealMatrix target) {

        /* input and target shuold be column vector like example
         * [1]
         * [0]
         * [1]
         */

        if (logger) {
            System.out.println("Original input parameters:");
            mathUtils.printMatrix(input);

            System.out.println("Original target parameters:");
            mathUtils.printMatrix(target);
        }

        // TODO scale input params to [0 ; 1]
        // (input[i] / 250.0 * 0.99) + 0.01

        // Calculate output signals from one layer to another
        // Need to multiply weights matrix on input parameters
        // Repeat for each layer. If NN has 3 leyers(input, hidden, output) need:
        // 1) wih * input = output_hidden; wih - weights matrix between input layer and hidden
        // 2) who * output_hidden = resultVector; who - weights matrix between hidden layer and output
        // 3) continue if NN has more than 3 layers.

        // iterationInput - represents input for each layer. on the begining its equals to input params
        // on the next iteration equals to calculated and activated signals after activation function
        RealMatrix iterationInput = input.copy();
        List<RealMatrix> activatedSignals = new ArrayList<>();
        activatedSignals.add(iterationInput);
        for (RealMatrix weightingFactor : weightingFactors) {

            mathUtils.printMatrix(weightingFactor);
            mathUtils.printMatrix(iterationInput);

            RealMatrix output = weightingFactor.multiply(iterationInput);

            mathUtils.printMatrix(output);

            // Apply activation function to each element in matrix
            iterationInput = activateNeurons(output);

            mathUtils.printMatrix(iterationInput);

            //Save calculated signals to calculate error on each layed
            activatedSignals.add(iterationInput);
        }

        if (logger) {
            System.out.println("Calculated output signal: ");
            mathUtils.printMatrix(iterationInput);
        }

        if (logger) {
            System.out.println("All calculated output signals: ");
            mathUtils.printMatrix(activatedSignals);
        }

        // Need to calculate output error for final layer and signals
        // Need to target_relust - result_signals(from last layer)
        // Example:
        // target_relust = [1, 0 ,1]
        // result_signals = [0,012, -1,056, 0,334] - calcalated by NN
        // [1, 0 ,1] - [0,012, -1,056, 0,334] = [0.988, -1.056, 0.666]
        // outputErrors will be beginig point to calculate athother error between hidden-output layer,
        // hidden-hidden layer and so forth
        RealMatrix lastLayerErrors = target.subtract(activatedSignals.get(activatedSignals.size() - 1));

        if (logger) {
            System.out.println("Calculated output errors. target - outputSignal");
            mathUtils.printMatrix(lastLayerErrors);
        }

        // As we got outputErrors on previuos step need to calcalute errors for hidden layer(s)
        // the formula is: hidden_errors = who' * outputErrors
        // Where `Who` - transpose weights matrix between hidden layer and output layer
        // When we get hidden_errors we can calculate errors for another hidden layers
        // hidden_hidden_errors = whh' * hidden_errors
        // and so forth

        // Out start point it errors for output layer
        RealMatrix iterationErrors = lastLayerErrors.copy();
        List<RealMatrix> errors = new ArrayList<>();
        errors.add(lastLayerErrors.copy());
        for (int i = weightingFactors.size() - 1; i > 0; i--) {
            RealMatrix weightingFactor = weightingFactors.get(i);

            mathUtils.printMatrix(weightingFactor.transpose());

            RealMatrix layerErrors = weightingFactor.transpose().multiply(iterationErrors);
            iterationErrors = layerErrors;

            mathUtils.printMatrix(layerErrors);

            errors.add(layerErrors);
        }

        if (logger) {
            System.out.println("Calculated errors: ");
            mathUtils.printMatrix(errors);
        }

        // Calculate back propagation of an error

//        if (errors.size() != activatedSignals.size()) {
//            throw new RuntimeException("Cout of errors size not equals to output signals size");
//        }

        int count = errors.size();
        int iteration = count;
        int activeSignalsCount = count + 1;

        // (output_errors * final_outputs * (1.0 - final_outputs) using scalar vector multiplication
        // self.who += numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        // self.wih += numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        for (int i = 0; i < count; i++) {
            RealMatrix layerErrors = errors.get(i);
            System.out.println("outputErrors");
            mathUtils.printMatrix(layerErrors);

            RealMatrix layerOutputSignal = activatedSignals.get(activeSignalsCount - 1);
            System.out.println("outputSignals");
            mathUtils.printMatrix(layerOutputSignal);

            // 1 - as
            RealMatrix oneMinusLayerOutputSignal = matrixMinusNumber(1.0, layerOutputSignal);
            System.out.println("1 - outputErrors");
            mathUtils.printMatrix(oneMinusLayerOutputSignal);

            RealMatrix pverlayerOutput = activatedSignals.get(activeSignalsCount - 2).transpose();
            System.out.println("input");
            mathUtils.printMatrix(pverlayerOutput);

            RealMatrix weightMatrix = weightingFactors.get(iteration - 1);
            System.out.println("weightMatrix");
            mathUtils.printMatrix(weightMatrix);

            // (output_errors * final_outputs * (1.0 - final_outputs)
            RealMatrix matrix = MatrixUtils.createColumnRealMatrix(
                    layerErrors.getColumnVector(0)
                            .ebeMultiply(
                                    layerOutputSignal
                                            .getColumnVector(0)
                            ).ebeMultiply(
                            oneMinusLayerOutputSignal
                                    .getColumnVector(0)
                    ).toArray());

            mathUtils.printMatrix(matrix);

            RealMatrix toUpdate = matrix.multiply(pverlayerOutput).scalarMultiply(learningRate);
            mathUtils.printMatrix(toUpdate);

            // Update weights
            // TODO: update matrix in list. matrix not updated
            weightMatrix = weightMatrix.add(toUpdate);
            mathUtils.printMatrix(weightMatrix);

            iteration--;
            activeSignalsCount--;
        }

        System.out.println("");

        mathUtils.printMatrix(weightingFactors);
    }


    @Override
    public RealMatrix doMagic(RealMatrix input) {
        if (logger) {
            System.out.println("Original input parameters:");
            mathUtils.printMatrix(input);
        }

        RealMatrix inputOnIteration = input.copy();

        for (RealMatrix weightingFactor : weightingFactors) {
            RealMatrix output = weightingFactor.multiply(inputOnIteration);
            inputOnIteration = activateNeurons(output);
        }

        if (logger) {
            System.out.println("Calculated output signals: ");
            mathUtils.printMatrix(inputOnIteration);
        }

        return inputOnIteration;
    }

    private void generateFactors() {
        for (int i = 0; i < layers.size() - 1; i++) {
            RealMatrix weightingMatrix = MatrixUtils.createRealMatrix(layers.get(i + 1).getNeuronsCount(), layers.get(i).getNeuronsCount());
            mathUtils.fillMatrixWithGaussian(weightingMatrix);
            weightingFactors.add(weightingMatrix);

            if (logger) {
                System.out.println(String.format("Generated weighting factors between layers %s and %s.", layers.get(i).getName(), layers.get(i + 1).getName()));
                mathUtils.printMatrix(weightingMatrix);
            }
        }
    }

    private RealMatrix activateNeurons(RealMatrix matrix) {
        double[][] data = matrix.getData();

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                double v = data[i][j];
                matrix.setEntry(i, j, activationFunction.activate(v));
            }
        }
        return matrix;
    }

    private RealMatrix matrixMinusNumber(double number, RealMatrix matrix) {
        double[][] data = matrix.getData();

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                double v = data[i][j];
                data[i][j] = number - v;
            }
        }
        return MatrixUtils.createRealMatrix(data);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public List<RealMatrix> getWeightingFactors() {
        return weightingFactors;
    }

    public void setWeightingFactors(List<RealMatrix> weightingFactors) {
        this.weightingFactors = weightingFactors;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public boolean isLogger() {
        return logger;
    }

    public void setLogger(boolean logger) {
        this.logger = logger;
    }
}
