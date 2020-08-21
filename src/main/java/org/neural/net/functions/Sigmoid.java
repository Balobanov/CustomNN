package org.neural.net.functions;

import org.apache.commons.math3.util.Precision;

public class Sigmoid implements ActivationFunction {

    @Override
    public double activate(double x) {
        double e = 2.7182818284;
        return Precision.round(1.0 / (1 + Math.pow(e, -(x))), 8);
    }
}
