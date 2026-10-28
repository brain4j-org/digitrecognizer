package it.echo.digitrecognizer.ui;

import java.awt.BasicStroke;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.util.Arrays;
import java.util.Locale;

import javax.swing.JComponent;
import javax.swing.Timer;

public final class PredictionPanel extends JComponent {

    private static final int SEGMENTS = 20;

    private final double[] display = new double[10];
    private final double[] target = new double[10];
    private final Timer animation;

    private int top = -1;
    private boolean busy;

    public PredictionPanel() {
        setOpaque(false);
        setPreferredSize(new java.awt.Dimension(320, 190));

        animation = new Timer(16, event -> animate());
        animation.setCoalesce(true);
    }

    public void setPredictions(float[] probabilities) {
        busy = false;
        top = argmax(probabilities);

        for (int i = 0; i < target.length; i++) {
            target[i] = probabilities[i];
        }

        if (!animation.isRunning()) {
            animation.start();
        }
        repaint();
    }

    public void setBusy(boolean busy) {
        this.busy = busy;
        repaint();
    }

    public void clear() {
        busy = false;
        top = -1;
        Arrays.fill(target, 0);
        if (!animation.isRunning()) {
            animation.start();
        }
        repaint();
    }

    private void animate() {
        boolean moving = false;

        for (int i = 0; i < display.length; i++) {
            double difference = target[i] - display[i];
            if (Math.abs(difference) > 0.0008) {
                display[i] += difference * 0.20;
                moving = true;
            } else {
                display[i] = target[i];
            }
        }

        if (!moving) {
            animation.stop();
        }

        repaint();
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        Graphics2D g = (Graphics2D) graphics.create();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        int padding = 22;
        int width = getWidth();
        int height = getHeight();
        int rightX = Math.max(126, Math.round(width * 0.42f));

        g.setFont(Theme.mono(Font.BOLD, 10.5f));
        g.setColor(Theme.MUTED);
        g.drawString("/ Prediction", padding, padding + 9);

        double confidence = top >= 0 ? Math.max(0, Math.min(1, display[top])) : 0;
        String digit = top < 0 ? "—" : Integer.toString(top);

        g.setFont(Theme.mono(Font.BOLD, 88f));
        g.setColor(top < 0 ? Theme.alpha(Theme.INK, 22) : Theme.ACCENT);
        FontMetrics digitMetrics = g.getFontMetrics();
        g.drawString(digit, padding, height - 35);

        g.setColor(Theme.alpha(Theme.INK, 35));
        g.drawLine(rightX - 16, padding + 24, rightX - 16, height - padding - 10);

        g.setFont(Theme.mono(Font.PLAIN, 10f));
        g.setColor(Theme.MUTED);
        g.drawString("/ Confidence", rightX, padding + 48);

        g.setFont(Theme.mono(Font.BOLD, 26f));
        g.setColor(top < 0 ? Theme.FAINT : Theme.INK);
        String percentage = String.format(Locale.US, "%.1f%%", confidence * 100);
        g.drawString(percentage, rightX, padding + 78);

        int meterX = rightX;
        int meterY = padding + 91;
        int meterWidth = width - rightX - padding;
        int gap = 2;
        float segmentWidth = (meterWidth - gap * (SEGMENTS - 1)) / (float) SEGMENTS;
        int filled = (int) Math.round(confidence * SEGMENTS);

        for (int i = 0; i < SEGMENTS; i++) {
            int x = Math.round(meterX + i * (segmentWidth + gap));
            int w = Math.max(1, Math.round(segmentWidth));
            g.setColor(i < filled ? Theme.ACCENT : Theme.alpha(Theme.INK, 28));
            g.fillRect(x, meterY, w, 6);
        }

        String status = busy ? "Analyzing" : top < 0 ? "Waiting" : null;
        if (status != null) {
            g.setFont(Theme.mono(Font.PLAIN, 10.5f));
            g.setColor(Theme.alpha(Theme.MUTED, 190));
            g.drawString(status, rightX, height - padding - 8);
        }

        if (top >= 0) {
            g.setStroke(new BasicStroke(1f));
            g.setColor(Theme.alpha(Theme.ACCENT, 80));
            g.drawLine(padding, height - 24, padding + digitMetrics.stringWidth(digit), height - 24);
        }

        g.dispose();
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
}
