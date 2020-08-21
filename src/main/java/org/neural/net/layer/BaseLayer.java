package org.neural.net.layer;

public class BaseLayer implements Layer {

    private final String name;
    private final int neurons;

    public BaseLayer(String name, int neurons) {
        this.name = name;
        this.neurons = neurons;
    }

    @Override
    public int getNeuronsCount() {
        return this.neurons;
    }

    @Override
    public String getName() {
        return this.name;
    }
}
