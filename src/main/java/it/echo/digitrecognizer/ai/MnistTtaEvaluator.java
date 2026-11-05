package it.echo.digitrecognizer.ai;

import java.util.List;

import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;

/**
 * Evaluates a model by averaging predictions over the shared test-time transforms.
 * Use it as a cheap robustness check before enabling test-time augmentation in the app.
 */
public final class MnistTtaEvaluator {

    private static final int BATCH_SIZE = 256;
    private static final int SIZE = MnistCnn.IMAGE_SIZE;

    private MnistTtaEvaluator() {
    }

    /**
     * Computes accuracy by averaging the probabilities of all augmented variants.
     *
     * @param model the trained model
     * @param samples the labelled samples
     * @return the accuracy in the range [0, 1]
     */
    public static double accuracy(Sequential model, List<Sample> samples) {
        List<MnistTransforms.Transform> transforms = MnistTransforms.testTimeVariants();
        int correct = 0;

        for (int start = 0; start < samples.size(); start += BATCH_SIZE) {
            int end = Math.min(start + BATCH_SIZE, samples.size());
            int batch = end - start;
            float[][] probabilities = new float[batch][MnistCnn.CLASSES];

            for (MnistTransforms.Transform transform : transforms) {
                accumulate(model, samples, start, end, transform, probabilities);
            }

            for (int i = 0; i < batch; i++) {
                if (argmax(probabilities[i]) == label(samples.get(start + i))) {
                    correct++;
                }
            }
        }

        return correct / (double) samples.size();
    }

    private static void accumulate(
        Sequential model,
        List<Sample> samples,
        int start,
        int end,
        MnistTransforms.Transform transform,
        float[][] probabilities
    ) {
        int batch = end - start;
        float[] flat = new float[batch * SIZE * SIZE];

        for (int i = start; i < end; i++) {
            float[] transformed = transform.apply(samples.get(i).input(0).data());
            System.arraycopy(transformed, 0, flat, (i - start) * SIZE * SIZE, SIZE * SIZE);
        }

        Tensor input = Tensors.create(new int[] {batch, 1, SIZE, SIZE}, flat);
        float[] output = model.predict(input).data();

        for (int i = 0; i < batch; i++) {
            for (int c = 0; c < MnistCnn.CLASSES; c++) {
                probabilities[i][c] += output[i * MnistCnn.CLASSES + c];
            }
        }
    }

    private static int argmax(float[] values) {
        int best = 0;

        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }

        return best;
    }

    private static int label(Sample sample) {
        return argmax(sample.label(0).data());
    }
}
