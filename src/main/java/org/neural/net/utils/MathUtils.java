package org.neural.net.utils;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MathUtils {

    private Random random;

    public MathUtils() {
        this.random = new Random();
    }

    public MathUtils(Random random) {
        this.random = random;
    }

    public RealMatrix fillMatrixWithGaussian(RealMatrix matrix) {

        double[][] data = matrix.getData();

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                matrix.setEntry(i, j, random.nextGaussian());
            }
        }

        return matrix;
    }

    public void printMatrix(RealMatrix matrix) {
        System.out.println("Begin of Matrix:");

        double[][] data = matrix.getData();
        for (double[] datum : data) {
            System.out.println(Arrays.toString(datum));
        }

        System.out.println("End of Matrix\n");
    }

    public void printMatrix(List<RealMatrix> realMatrices) {
        realMatrices.forEach(this::printMatrix);
    }

    public void scalarAdding(double number, RealMatrix matrix) {

    }
}
