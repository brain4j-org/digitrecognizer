package it.echo.digitrecognizer.ai;

import java.util.ArrayList;
import java.util.List;

/**
 * Builds deterministic transforms used for test-time augmentation.
 * Use the same variants during evaluation and in the application so metrics stay comparable.
 */
public final class MnistTransforms {

    private static final int SIZE = MnistCnn.IMAGE_SIZE;

    /**
     * Applies one deterministic transformation to a 28x28 digit.
     */
    public interface Transform {

        /**
         * Returns the transformed pixels.
         *
         * @param source the 28x28 pixel values
         * @return the transformed pixels
         */
        float[] apply(float[] source);
    }

    private MnistTransforms() {
    }

    /**
     * Returns the default test-time variants: identity, small shifts and small scales.
     *
     * @return the transform list
     */
    public static List<Transform> testTimeVariants() {
        List<Transform> transforms = new ArrayList<>();
        transforms.add(source -> source);

        for (int[] shift : new int[][] {{-2, 0}, {2, 0}, {0, -2}, {0, 2}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
            transforms.add(source -> shift(source, shift[0], shift[1]));
        }

        for (double scale : new double[] {0.95, 1.05}) {
            transforms.add(source -> scale(source, scale));
        }

        return List.copyOf(transforms);
    }

    /**
     * Translates the digit, filling the exposed border with zeros.
     *
     * @param source the 28x28 pixel values
     * @param shiftX horizontal shift in pixels
     * @param shiftY vertical shift in pixels
     * @return the shifted pixels
     */
    public static float[] shift(float[] source, int shiftX, int shiftY) {
        float[] result = new float[SIZE * SIZE];

        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                int sourceX = x - shiftX;
                int sourceY = y - shiftY;

                if (sourceX < 0 || sourceY < 0 || sourceX >= SIZE || sourceY >= SIZE) {
                    continue;
                }

                result[y * SIZE + x] = source[sourceY * SIZE + sourceX];
            }
        }

        return result;
    }

    /**
     * Scales the digit around the image center.
     *
     * @param source the 28x28 pixel values
     * @param factor the scale factor
     * @return the scaled pixels
     */
    public static float[] scale(float[] source, double factor) {
        float[] result = new float[SIZE * SIZE];
        double center = (SIZE - 1) / 2.0;

        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                double sourceX = (x - center) / factor + center;
                double sourceY = (y - center) / factor + center;

                result[y * SIZE + x] = sample(source, sourceX, sourceY);
            }
        }

        return result;
    }

    private static float sample(float[] source, double x, double y) {
        if (x < 0 || y < 0 || x > SIZE - 1 || y > SIZE - 1) {
            return 0f;
        }

        int x0 = Math.min((int) x, SIZE - 1);
        int y0 = Math.min((int) y, SIZE - 1);
        int x1 = Math.min(x0 + 1, SIZE - 1);
        int y1 = Math.min(y0 + 1, SIZE - 1);
        double tx = x - x0;
        double ty = y - y0;

        double top = source[y0 * SIZE + x0] * (1 - tx) + source[y0 * SIZE + x1] * tx;
        double bottom = source[y1 * SIZE + x0] * (1 - tx) + source[y1 * SIZE + x1] * tx;

        return (float) (top * (1 - ty) + bottom * ty);
    }
}
