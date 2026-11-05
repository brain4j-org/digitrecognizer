package it.echo.digitrecognizer.ai;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.ConvLayer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.MaxPoolLayer;
import org.brain4j.core.layer.impl.ReshapeLayer;
import org.brain4j.core.model.ModelLayout;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.weightsinit.impl.UniformHeInit;

public final class MnistCnn {

    public static final int IMAGE_SIZE = 28;
    public static final int CLASSES = 10;

    private MnistCnn() {
    }

    /**
     * Creates the default network.
     *
     * @return a compiled model
     */
    public static Sequential create() {
        return create(0.25);
    }

    /**
     * Creates the network with the given dropout rate.
     *
     * @param dropoutRate dropout probability before the output layer
     * @return a compiled model
     */
    public static Sequential create(double dropoutRate) {
        return ModelLayout.of(Shape.of(1, IMAGE_SIZE, IMAGE_SIZE),
            he(new ConvLayer(32, 3, 3, 1, new ReLU())),
            new MaxPoolLayer(2, 2, 2),
            he(new ConvLayer(64, 3, 3, 1, new ReLU())),
            new MaxPoolLayer(2, 2, 2),
            new ReshapeLayer(Shape.of(64 * 5 * 5)),
            he(new DenseLayer(128, new ReLU())),
            new DropoutLayer(dropoutRate),
            new DenseLayer(CLASSES, new Softmax())
        ).compile();
    }

    /**
     * Creates a deeper network with four convolutional layers.
     *
     * @param dropoutRate dropout probability before the output layer
     * @return a compiled model
     */
    public static Sequential createLarge(double dropoutRate) {
        return ModelLayout.of(Shape.of(1, IMAGE_SIZE, IMAGE_SIZE),
            he(new ConvLayer(32, 3, 3, 1, new ReLU())),
            he(new ConvLayer(32, 3, 3, 1, new ReLU())),
            new MaxPoolLayer(2, 2, 2),
            he(new ConvLayer(64, 3, 3, 1, new ReLU())),
            he(new ConvLayer(64, 3, 3, 1, new ReLU())),
            new MaxPoolLayer(2, 2, 2),
            new ReshapeLayer(Shape.of(64 * 4 * 4)),
            he(new DenseLayer(128, new ReLU())),
            new DropoutLayer(dropoutRate),
            new DenseLayer(CLASSES, new Softmax())
        ).compile();
    }

    /**
     * Creates a wider network with more filters and a larger dense layer.
     *
     * @param dropoutRate dropout probability before the output layer
     * @return a compiled model
     */
    public static Sequential createWide(double dropoutRate) {
        return ModelLayout.of(Shape.of(1, IMAGE_SIZE, IMAGE_SIZE),
            he(new ConvLayer(48, 3, 3, 1, new ReLU())),
            new MaxPoolLayer(2, 2, 2),
            he(new ConvLayer(96, 3, 3, 1, new ReLU())),
            new MaxPoolLayer(2, 2, 2),
            new ReshapeLayer(Shape.of(96 * 5 * 5)),
            he(new DenseLayer(192, new ReLU())),
            new DropoutLayer(dropoutRate),
            new DenseLayer(CLASSES, new Softmax())
        ).compile();
    }

    /**
     * Creates a network with two dense layers after the convolutions.
     *
     * @param dropoutRate dropout probability before each dense transition
     * @return a compiled model
     */
    public static Sequential createDense2(double dropoutRate) {
        return ModelLayout.of(Shape.of(1, IMAGE_SIZE, IMAGE_SIZE),
            he(new ConvLayer(32, 3, 3, 1, new ReLU())),
            new MaxPoolLayer(2, 2, 2),
            he(new ConvLayer(64, 3, 3, 1, new ReLU())),
            new MaxPoolLayer(2, 2, 2),
            new ReshapeLayer(Shape.of(64 * 5 * 5)),
            he(new DenseLayer(256, new ReLU())),
            new DropoutLayer(dropoutRate),
            he(new DenseLayer(128, new ReLU())),
            new DropoutLayer(dropoutRate),
            new DenseLayer(CLASSES, new Softmax())
        ).compile();
    }

    private static Layer he(Layer layer) {
        return layer.weightInit(new UniformHeInit());
    }
}
