package it.echo.digitrecognizer.ai;

import java.util.SplittableRandom;

/**
 * Applies random affine and elastic deformations to a 28x28 MNIST digit.
 * Use one instance per training run so that the seed is reproducible.
 */
public final class MnistAugmenter {

    private static final int SIZE = 28;
    private static final double CENTER = (SIZE - 1) / 2.0;
    private static final int ELASTIC_GRID = 4;

    private final SplittableRandom random;
    private final double maxAngle;
    private final double maxShear;
    private final double maxScale;
    private final double elasticStrength;
    private final double strokeChance;

    /**
     * Creates an augmenter with the default deformation ranges.
     *
     * @param seed the random seed
     */
    public MnistAugmenter(long seed) {
        this(seed, 12.0, 0.12, 0.08, 1.5, 0.35);
    }

    /**
     * Creates an augmenter with explicit deformation ranges.
     *
     * @param seed the random seed
     * @param maxAngle maximum absolute rotation in degrees
     * @param maxShear maximum absolute horizontal shear
     * @param maxScale maximum absolute scale variation
     * @param elasticStrength maximum elastic displacement in pixels
     * @param strokeChance probability of applying a thickness change
     */
    public MnistAugmenter(
        long seed,
        double maxAngle,
        double maxShear,
        double maxScale,
        double elasticStrength,
        double strokeChance
    ) {
        this.random = new SplittableRandom(seed);
        this.maxAngle = maxAngle;
        this.maxShear = maxShear;
        this.maxScale = maxScale;
        this.elasticStrength = elasticStrength;
        this.strokeChance = strokeChance;
    }

    /**
     * Returns a randomly deformed copy of the given digit.
     *
     * @param source the 28x28 pixel values
     * @return the deformed pixels
     */
    public float[] augment(float[] source) {
        float[] result = new float[SIZE * SIZE];

        double angle = Math.toRadians(random.nextDouble(-maxAngle, maxAngle));
        double shear = random.nextDouble(-maxShear, maxShear);
        double scale = 1.0 + random.nextDouble(-maxScale, maxScale);
        double cosine = Math.cos(angle);
        double sine = Math.sin(angle);
        double inverseScale = 1.0 / scale;

        double[] elasticX = elasticField();
        double[] elasticY = elasticField();

        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                double outputX = x - CENTER;
                double outputY = y - CENTER;

                double rotatedX = (cosine * outputX + sine * outputY) * inverseScale;
                double rotatedY = (-sine * outputX + cosine * outputY) * inverseScale;

                double sourceX = rotatedX - shear * rotatedY;
                double sourceY = rotatedY;

                sourceX += elasticAt(elasticX, sourceX, sourceY);
                sourceY += elasticAt(elasticY, sourceX, sourceY);

                result[y * SIZE + x] = sample(source, sourceX + CENTER, sourceY + CENTER);
            }
        }

        if (random.nextDouble() < strokeChance) {
            result = changeThickness(result, random.nextBoolean());
        }

        return result;
    }

    private double[] elasticField() {
        double[] field = new double[ELASTIC_GRID * ELASTIC_GRID];

        for (int i = 0; i < field.length; i++) {
            field[i] = random.nextDouble(-elasticStrength, elasticStrength);
        }

        return field;
    }

    private double elasticAt(double[] field, double x, double y) {
        double gridX = (x + CENTER) / (SIZE - 1) * (ELASTIC_GRID - 1);
        double gridY = (y + CENTER) / (SIZE - 1) * (ELASTIC_GRID - 1);

        int x0 = clamp((int) Math.floor(gridX), ELASTIC_GRID - 1);
        int y0 = clamp((int) Math.floor(gridY), ELASTIC_GRID - 1);
        int x1 = Math.min(x0 + 1, ELASTIC_GRID - 1);
        int y1 = Math.min(y0 + 1, ELASTIC_GRID - 1);

        double tx = gridX - x0;
        double ty = gridY - y0;

        double top = field[y0 * ELASTIC_GRID + x0] * (1 - tx)
            + field[y0 * ELASTIC_GRID + x1] * tx;
        double bottom = field[y1 * ELASTIC_GRID + x0] * (1 - tx)
            + field[y1 * ELASTIC_GRID + x1] * tx;

        return top * (1 - ty) + bottom * ty;
    }

    private float sample(float[] source, double x, double y) {
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

    private float[] changeThickness(float[] source, boolean dilate) {
        float[] result = new float[SIZE * SIZE];

        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                float value = source[y * SIZE + x];

                for (int offsetY = -1; offsetY <= 1; offsetY++) {
                    for (int offsetX = -1; offsetX <= 1; offsetX++) {
                        int sampleY = y + offsetY;
                        int sampleX = x + offsetX;

                        if (sampleX < 0 || sampleY < 0 || sampleX >= SIZE || sampleY >= SIZE) {
                            continue;
                        }

                        float neighbour = source[sampleY * SIZE + sampleX];
                        value = dilate ? Math.max(value, neighbour) : Math.min(value, neighbour);
                    }
                }

                result[y * SIZE + x] = value;
            }
        }

        return result;
    }

    private static int clamp(int value, int max) {
        return Math.max(0, Math.min(max, value));
    }
}
