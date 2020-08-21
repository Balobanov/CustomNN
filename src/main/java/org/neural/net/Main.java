package org.neural.net;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.neural.net.core.NeuralNetwork;
import org.neural.net.layer.BaseLayer;
import org.neural.net.utils.MathUtils;

import java.util.List;

public class Main {

    public static MathUtils mathUtils = new MathUtils();

    public static void main(String[] args) {
        double[][] wih = new double[][]{
                {-0.41129894, 0.43518722, -0.02569386},
                {0.26085398, 0.77659483, 0.30734542},
                {0.77953135, 0.49722062, 0.85371961}
        };

        double[][] who = new double[][]{
                {-0.60354877, -0.45552303, -0.72838853},
                {0.32495974, -0.14048448, 0.52754844},
                {0.18322264, 0.07349858, 1.24152419}
        };

        List<RealMatrix> w = List.of(
                MatrixUtils.createRealMatrix(wih),
                MatrixUtils.createRealMatrix(who)
        );

//        mathUtils.printMatrix(w);

        NeuralNetwork neuralNetwork = new NeuralNetwork(List.of(
                new BaseLayer("Input", 3),
                new BaseLayer("Hidden_1", 3),
                new BaseLayer("Output", 3)
        ), w);

//        NeuralNetwork neuralNetwork = new NeuralNetwork(List.of(
//                new BaseLayer("Input", 3),
//                new BaseLayer("Hidden_1", 8),
//                new BaseLayer("Hidden_2", 8),
//                new BaseLayer("Hidden_3", 8),
//                new BaseLayer("Hidden_4", 8),
//                new BaseLayer("Output", 3)
//        ));

        neuralNetwork.setLogger(true);
        neuralNetwork.train(
                MatrixUtils.createColumnRealMatrix(new double[]{1.0, 0.0, 1.0}),
                MatrixUtils.createColumnRealMatrix(new double[]{0.0, 1.0, 0.0})
        );
    }
}
