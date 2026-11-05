package it.echo.digitrecognizer.ai;

import org.brain4j.core.training.optimizer.impl.AdamW;

/**
 * AdamW whose learning rate can be updated between epochs.
 * Use it to apply an external learning-rate schedule without resetting optimizer state.
 */
public final class AdjustableAdamW extends AdamW {

    /**
     * Creates an optimizer with the given initial learning rate and weight decay.
     *
     * @param learningRate the initial learning rate
     * @param weightDecay the decoupled weight decay
     */
    public AdjustableAdamW(double learningRate, double weightDecay) {
        super(learningRate, weightDecay);
    }

    /**
     * Replaces the learning rate used by subsequent updates.
     *
     * @param learningRate the new learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
