package it.echo.digitrecognizer.ai;

import java.util.IdentityHashMap;
import java.util.Map;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.tensor.Tensor;

/**
 * Averages model parameters collected at the end of several epochs.
 * Use it for stochastic weight averaging without keeping full model copies.
 */
public final class WeightAverager {

    private final Map<Tensor, float[]> sums = new IdentityHashMap<>();
    private int snapshots;

    /**
     * Adds the current model parameters to the running average.
     *
     * @param model the model to snapshot
     */
    public void add(Sequential model) {
        for (Layer layer : model.layers()) {
            for (Tensor parameter : layer.parameters().values()) {
                float[] sum = sums.computeIfAbsent(parameter, key -> new float[key.elements()]);
                float[] data = parameter.data();

                for (int i = 0; i < data.length; i++) {
                    sum[i] += data[i];
                }
            }
        }

        snapshots++;
    }

    /**
     * Replaces the model parameters with their average.
     *
     * @param model the model to update
     */
    public void applyTo(Sequential model) {
        if (snapshots == 0) {
            return;
        }

        for (Layer layer : model.layers()) {
            for (Tensor parameter : layer.parameters().values()) {
                float[] sum = sums.get(parameter);

                if (sum == null) {
                    continue;
                }

                float[] data = parameter.data();
                float inverse = 1f / snapshots;

                for (int i = 0; i < data.length; i++) {
                    data[i] = sum[i] * inverse;
                }
            }
        }
    }

    /**
     * Returns the number of collected snapshots.
     *
     * @return the snapshot count
     */
    public int snapshots() {
        return snapshots;
    }
}
