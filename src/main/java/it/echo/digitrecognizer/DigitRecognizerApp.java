package it.echo.digitrecognizer;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import it.echo.digitrecognizer.ui.Theme;

public final class DigitRecognizerApp {

    private DigitRecognizerApp() {
    }

    public static void main(String[] args) {
        System.setProperty("awt.useSystemAAFontSettings", "on");
        System.setProperty("swing.aatext", "true");

        Theme.install();
        JFrame.setDefaultLookAndFeelDecorated(true);

        SwingUtilities.invokeLater(() -> {
            MainFrame frame = new MainFrame();
            frame.setVisible(true);
        });
    }
}
