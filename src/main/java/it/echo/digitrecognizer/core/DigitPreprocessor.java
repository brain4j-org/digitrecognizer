package it.echo.digitrecognizer.core;

public final class DigitPreprocessor {

    public static final int SIZE = 28;
    public static final int CONTENT_SIZE = 20;
    private static final float INK_THRESHOLD = 0.06f;

    private DigitPreprocessor() {
    }

    public static float[] prepare(float[] grid) {
        if (grid == null || grid.length != SIZE * SIZE) {
            throw new IllegalArgumentException("Expected a 28x28 grid");
        }

        int minX = SIZE;
        int minY = SIZE;
        int maxX = -1;
        int maxY = -1;

        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                if (grid[y * SIZE + x] > INK_THRESHOLD) {
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        if (maxX < minX || maxY < minY) {
            return new float[SIZE * SIZE];
        }

        int sourceWidth = maxX - minX + 1;
        int sourceHeight = maxY - minY + 1;

        float scale = CONTENT_SIZE / (float) Math.max(sourceWidth, sourceHeight);
        int targetWidth = Math.max(1, Math.round(sourceWidth * scale));
        int targetHeight = Math.max(1, Math.round(sourceHeight * scale));

        float[] scaled = resize(grid, minX, minY, sourceWidth, sourceHeight, targetWidth, targetHeight);

        float peak = 0f;
        for (float value : scaled) {
            peak = Math.max(peak, value);
        }
        if (peak > 1e-6f) {
            float normalization = 1f / peak;
            for (int i = 0; i < scaled.length; i++) {
                scaled[i] *= normalization;
            }
        }

        double mass = 0;
        double massX = 0;
        double massY = 0;

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                float value = scaled[y * targetWidth + x];
                mass += value;
                massX += x * value;
                massY += y * value;
            }
        }

        int offsetX = 0;
        int offsetY = 0;

        if (mass > 0) {
            offsetX = Math.round(SIZE / 2f - (float) (massX / mass));
            offsetY = Math.round(SIZE / 2f - (float) (massY / mass));
        }

        offsetX = Math.max(0, Math.min(SIZE - targetWidth, offsetX));
        offsetY = Math.max(0, Math.min(SIZE - targetHeight, offsetY));

        float[] result = new float[SIZE * SIZE];

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                result[(y + offsetY) * SIZE + x + offsetX] = Math.min(1f, scaled[y * targetWidth + x]);
            }
        }

        return result;
    }

    private static float[] resize(
        float[] source,
        int sourceX,
        int sourceY,
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight
    ) {
        float[] result = new float[targetWidth * targetHeight];
        double scaleX = sourceWidth / (double) targetWidth;
        double scaleY = sourceHeight / (double) targetHeight;

        for (int ty = 0; ty < targetHeight; ty++) {
            double y0 = sourceY + ty * scaleY;
            double y1 = sourceY + (ty + 1) * scaleY;

            int firstY = (int) Math.floor(y0);
            int lastY = (int) Math.ceil(y1) - 1;

            for (int tx = 0; tx < targetWidth; tx++) {
                double x0 = sourceX + tx * scaleX;
                double x1 = sourceX + (tx + 1) * scaleX;

                int firstX = (int) Math.floor(x0);
                int lastX = (int) Math.ceil(x1) - 1;

                double weightedSum = 0;
                double totalWeight = 0;

                for (int sy = firstY; sy <= lastY; sy++) {
                    double overlapY = Math.min(y1, sy + 1) - Math.max(y0, sy);
                    if (overlapY <= 0 || sy < 0 || sy >= SIZE) {
                        continue;
                    }

                    for (int sx = firstX; sx <= lastX; sx++) {
                        double overlapX = Math.min(x1, sx + 1) - Math.max(x0, sx);
                        if (overlapX <= 0 || sx < 0 || sx >= SIZE) {
                            continue;
                        }

                        double weight = overlapX * overlapY;
                        weightedSum += source[sy * SIZE + sx] * weight;
                        totalWeight += weight;
                    }
                }

                result[ty * targetWidth + tx] = totalWeight > 0
                    ? (float) (weightedSum / totalWeight)
                    : 0f;
            }
        }

        return result;
    }
}
