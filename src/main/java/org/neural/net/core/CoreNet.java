package org.neural.net.core;

import org.apache.commons.math3.linear.RealMatrix;

public interface CoreNet {
    void train(RealMatrix input, RealMatrix target);
    RealMatrix doMagic(RealMatrix input);
}
