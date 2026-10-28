package it.echo.digitrecognizer.ai;

import org.brain4j.core.layer.impl.ConvLayer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.MaxPoolLayer;
import org.brain4j.core.layer.impl.ReshapeLayer;
import org.brain4j.core.model.ModelLayout;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.tensor.Shape;

public final class MnistCnn {

    public static final int IMAGE_SIZE = 28;
    public static final int CLASSES = 10;

    private MnistCnn() {
    }

    public static Sequential create() {
        return ModelLayout.of(Shape.of(1, IMAGE_SIZE, IMAGE_SIZE),
            new ConvLayer(32, 3, 3, 1, new ReLU()),
            new MaxPoolLayer(2, 2, 2),
            new ConvLayer(64, 3, 3, 1, new ReLU()),
            new MaxPoolLayer(2, 2, 2),
            new ReshapeLayer(Shape.of(64 * 5 * 5)),
            new DenseLayer(128, new ReLU()),
            new DenseLayer(CLASSES, new Softmax())
        ).compile();
    }
}
