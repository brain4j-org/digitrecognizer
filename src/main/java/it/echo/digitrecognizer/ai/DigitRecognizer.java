package it.echo.digitrecognizer.ai;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.brain4j.core.Brain4J;
import org.brain4j.core.importing.format.impl.BrainFormat;
import org.brain4j.core.importing.io.ModelIO;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

public final class DigitRecognizer implements AutoCloseable {

    public static final int CLASSES = 10;

    private static final String MODEL_RESOURCE = "/models/mnist-cnn.zip";
    private static final Path DEV_MODEL = Path.of("src/main/resources/models/mnist-cnn.zip");
    private static final List<MnistTransforms.Transform> TTA_TRANSFORMS =
        MnistTransforms.testTimeVariants();

    private Sequential model;
    private volatile Device device;
    private final ExecutorService executor;
    private final Object inferenceLock = new Object();

    private DigitRecognizer(Sequential model) {
        this.model = model;
        this.executor = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "digit-inference");
            thread.setDaemon(true);
            return thread;
        });
    }

    public static DigitRecognizer loadBundled() throws IOException {
        Path path = locateModel();
        Sequential model = ModelIO.load(new BrainFormat(), path.toFile());

        return new DigitRecognizer(model).warmUp();
    }

    private DigitRecognizer warmUp() {
        try {
            predict(new float[28 * 28]);
        } catch (RuntimeException exception) {
            System.err.println("Warm-up failed: " + rootMessage(exception));
        }

        return this;
    }

    public float[] predict(float[] pixels28x28) {
        if (pixels28x28.length != 28 * 28) {
            throw new IllegalArgumentException("Expected 784 pixels");
        }

        Tensor batch = ttaBatch(pixels28x28);

        synchronized (inferenceLock) {
            float[] output;

            if (device == null) {
                output = model.predict(batch).data();
            } else {
                device.createResources();

                try {
                    Tensor input = batch.to(device);
                    output = model.graph().predict(new StatesCache(false), input)[0].data();
                } finally {
                    device.closeResources();
                }
            }

            return average(output);
        }
    }

    private static Tensor ttaBatch(float[] pixels28x28) {
        int variants = TTA_TRANSFORMS.size();
        float[] flat = new float[variants * pixels28x28.length];

        for (int i = 0; i < variants; i++) {
            float[] transformed = TTA_TRANSFORMS.get(i).apply(pixels28x28);
            System.arraycopy(transformed, 0, flat, i * pixels28x28.length, pixels28x28.length);
        }

        return Tensors.create(new int[] {variants, 1, 28, 28}, flat);
    }

    private static float[] average(float[] output) {
        float[] probabilities = new float[CLASSES];
        int variants = TTA_TRANSFORMS.size();

        for (int i = 0; i < variants; i++) {
            for (int c = 0; c < CLASSES; c++) {
                probabilities[c] += output[i * CLASSES + c];
            }
        }

        float inverse = 1f / variants;

        for (int c = 0; c < CLASSES; c++) {
            probabilities[c] *= inverse;
        }

        return probabilities;
    }

    public CompletableFuture<float[]> predictAsync(float[] pixels28x28) {
        return CompletableFuture.supplyAsync(() -> predict(pixels28x28), executor);
    }

    public CompletableFuture<Void> useGpuAsync() {
        return CompletableFuture.runAsync(() -> {
            if (device != null) {
                return;
            }

            Device gpu;

            try {
                gpu = new Device(0);
            } catch (Throwable throwable) {
                throw new CompletionException(new IllegalStateException(
                    "GPU unavailable: " + rootMessage(throwable), throwable));
            }

            try {
                Brain4J.initKernels(gpu);

                synchronized (inferenceLock) {
                    model = model.to(gpu);
                    device = gpu;
                }
            } catch (Throwable throwable) {
                freeQuietly(gpu);
                throw new CompletionException(new IllegalStateException(
                    "GPU unavailable: " + rootMessage(throwable), throwable));
            }
        }, executor);
    }

    public CompletableFuture<Void> useCpuAsync() {
        return CompletableFuture.runAsync(() -> {
            if (device == null) {
                return;
            }

            synchronized (inferenceLock) {
                model = model.to(null);
                Device oldDevice = device;
                device = null;
                freeQuietly(oldDevice);
            }
        }, executor);
    }

    public boolean isUsingGpu() {
        return device != null;
    }

    public String deviceName() {
        Device current = device;
        return current == null ? "CPU" : current.name();
    }

    public String description() {
        int parameters = 0;

        for (var layer : model.layers()) {
            for (Tensor parameter : layer.parameters().values()) {
                parameters += parameter.elements();
            }
        }

        return "MNIST CNN · " + String.format(Locale.US, "%,d", parameters) + " parameters";
    }

    @Override
    public void close() {
        try {
            executor.execute(() -> {
                synchronized (inferenceLock) {
                    if (device != null) {
                        try {
                            model = model.to(null);
                        } finally {
                            freeQuietly(device);
                            device = null;
                        }
                    }
                }
            });
        } catch (RuntimeException ignored) {
            // The executor was already shut down.
        }

        executor.shutdown();
    }

    private static void freeQuietly(Device device) {
        if (device == null) {
            return;
        }

        try {
            device.free();
        } catch (RuntimeException ignored) {
            // Device release is best-effort.
        }
    }

    private static String rootMessage(Throwable throwable) {
        Throwable cause = throwable;

        while (cause.getCause() != null && cause.getCause() != cause) {
            cause = cause.getCause();
        }

        String message = cause.getMessage();
        return message == null || message.isBlank() ? cause.getClass().getSimpleName() : message;
    }

    private static Path locateModel() throws IOException {
        try (InputStream stream = DigitRecognizer.class.getResourceAsStream(MODEL_RESOURCE)) {
            if (stream == null) {
                if (Files.isRegularFile(DEV_MODEL)) {
                    return DEV_MODEL;
                }

                throw new IOException("Model not found. Run: ./gradlew train");
            }

            Path temp = Files.createTempFile("digitrecognizer-", ".zip");
            Files.copy(stream, temp, StandardCopyOption.REPLACE_EXISTING);
            temp.toFile().deleteOnExit();
            return temp;
        }
    }
}
