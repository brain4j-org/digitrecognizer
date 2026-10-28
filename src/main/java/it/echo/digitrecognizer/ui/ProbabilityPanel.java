package it.echo.digitrecognizer.ui;

import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.util.Arrays;
import java.util.Locale;

import javax.swing.JComponent;
import javax.swing.Timer;

public final class ProbabilityPanel extends JComponent {

    private final double[] display = new double[10];
    private final double[] target = new double[10];
    private final Timer animation;

    private int top = -1;

    public ProbabilityPanel() {
        setOpaque(false);
        setPreferredSize(new java.awt.Dimension(320, 390));

        animation = new Timer(16, event -> animate());
        animation.setCoalesce(true);
    }

    public void setPredictions(float[] probabilities) {
        top = argmax(probabilities);
        for (int i = 0; i < target.length; i++) {
            target[i] = probabilities[i];
        }
        if (!animation.isRunning()) {
            animation.start();
        }
        repaint();
    }

    public void clear() {
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

        g.setFont(Theme.mono(Font.BOLD, 10.5f));
        g.setColor(Theme.MUTED);
        g.drawString("/ Probabilities", padding, padding + 9);

        g.setFont(Theme.mono(Font.PLAIN, 10f));
        g.setColor(Theme.alpha(Theme.MUTED, 170));
        String range = "0 — 9";
        g.drawString(range, width - padding - g.getFontMetrics().stringWidth(range), padding + 9);

        int rowsTop = padding + 30;
        int rowsBottom = padding;
        int available = height - rowsTop - rowsBottom;
        int rowHeight = Math.min(38, Math.max(26, available / 10));
        int rowsHeight = rowHeight * 10;
        int extraSpace = Math.max(0, available - rowsHeight);
        rowsTop += Math.min(30, extraSpace / 2);

        int labelWidth = 24;
        int percentWidth = 55;
        int trackX = padding + labelWidth;
        int trackRight = width - padding - percentWidth;
        int trackWidth = Math.max(40, trackRight - trackX);

        g.setColor(Theme.alpha(Theme.INK, 30));
        g.drawLine(padding, rowsTop - 7, width - padding, rowsTop - 7);

        for (int digit = 0; digit < 10; digit++) {
            int rowY = rowsTop + digit * rowHeight;
            int centerY = rowY + rowHeight / 2;
            boolean isTop = digit == top && top >= 0;

            if (isTop) {
                g.setColor(Theme.ACCENT);
                g.fillRect(padding - 9, rowY + 7, 2, rowHeight - 14);
            }

            g.setColor(Theme.alpha(Theme.INK, 23));
            g.drawLine(padding, rowY + rowHeight - 1, width - padding, rowY + rowHeight - 1);

            g.setFont(Theme.mono(isTop ? Font.BOLD : Font.PLAIN, 13f));
            g.setColor(isTop ? Theme.INK : Theme.MUTED);
            g.drawString(Integer.toString(digit), padding, centerY + 4);

            g.setColor(Theme.alpha(Theme.INK, 24));
            g.fillRect(trackX, centerY - 1, trackWidth, 2);

            g.setColor(Theme.alpha(Theme.INK, 18));
            for (int tick = 1; tick < 4; tick++) {
                int tickX = trackX + trackWidth * tick / 4;
                g.fillRect(tickX, centerY - 4, 1, 3);
            }

            float value = (float) Math.max(0, Math.min(1, display[digit]));
            if (value > 0.001f) {
                g.setColor(isTop ? Theme.ACCENT : Theme.alpha(Theme.INK_SOFT, 155));
                int fillWidth = Math.max(2, Math.round(trackWidth * value));
                g.fillRect(trackX, centerY - 2, fillWidth, 4);
            }

            g.setFont(Theme.mono(isTop ? Font.BOLD : Font.PLAIN, 11f));
            g.setColor(isTop ? Theme.INK : Theme.alpha(Theme.MUTED, 190));
            String percentage = String.format(Locale.US, "%.1f%%", value * 100);
            g.drawString(percentage,
                width - padding - g.getFontMetrics().stringWidth(percentage),
                centerY + 4);
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
