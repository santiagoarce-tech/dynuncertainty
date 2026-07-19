"""Graphical catalogue for selecting dynuncertainty models."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from dynuncertainty.interface import InteractiveModelRunner
from dynuncertainty.models import (
    build_damped_oscillator_model,
    build_harmonic_oscillator_model,
    build_nonlinear_pendulum_model,
)


class ModelCatalogWindow:
    """Main window used to select and open dynamical models."""

    def __init__(self) -> None:
        self.figure = plt.figure(figsize=(9, 6))
        self.model_buttons: list[Button] = []
        self.open_runners: list[InteractiveModelRunner] = []

        window_manager = self.figure.canvas.manager

        if window_manager is not None:
            window_manager.set_window_title(
                "dynuncertainty — Model catalogue"
            )

        self._build_interface()

    def _build_interface(self) -> None:
        """Construct the visible catalogue entries."""

        self.figure.suptitle(
            "dynuncertainty — Model catalogue",
            fontsize=17,
            fontweight="bold",
        )

        self.figure.text(
            0.10,
            0.86,
            "Select a dynamical model to configure and simulate.",
            fontsize=11,
        )
        catalogue_entries = (
            (
                "Harmonic oscillator",
                "Explore uncertainty in ideal periodic motion.",
            ),
            (
                "Damped oscillator",
                "Explore uncertainty in dissipative oscillatory motion.",
            ),
            (
                "Nonlinear pendulum",
                "Explore uncertainty in nonlinear pendulum dynamics.",
            ),
        )

        y_positions = (0.68, 0.46, 0.24)

        for (model_name, description), y_position in zip(
            catalogue_entries,
            y_positions,
        ):
            self.figure.text(
                0.12,
                y_position + 0.045,
                model_name,
                fontsize=13,
                fontweight="bold",
            )

            self.figure.text(
                0.12,
                y_position,
                description,
                fontsize=10,
            )

            button_axis = self.figure.add_axes(
                (0.64, y_position - 0.01, 0.22, 0.075)
            )

            button = Button(
                button_axis,
                "Open model",
                color="lightsteelblue",
                hovercolor="cornflowerblue",
            )

            self.model_buttons.append(button)

        
        self.model_buttons[0].on_clicked(
            self._open_harmonic_oscillator
        )
       
        self.model_buttons[1].on_clicked(
            self._open_damped_oscillator
        )

        self.model_buttons[2].on_clicked(
            self._open_nonlinear_pendulum
        )

        
        self.status_text = self.figure.text(
            0.10,
            0.08,
            "Catalogue ready.",
            fontsize=10,
            color="darkgreen",
        )

    def _open_harmonic_oscillator(self, _event) -> None:
        """Open an independent harmonic-oscillator interface."""

        model = build_harmonic_oscillator_model()
        runner = InteractiveModelRunner(model)

        self.open_runners.append(runner)
        runner.figure.show()

        self.status_text.set_text(
            "Opened: Harmonic oscillator."
        )
        self.figure.canvas.draw_idle()

    def _open_damped_oscillator(self, _event) -> None:
        """Open an independent damped-oscillator interface."""

        model = build_damped_oscillator_model()
        runner = InteractiveModelRunner(model)

        self.open_runners.append(runner)
        runner.figure.show()

        self.status_text.set_text(
            "Opened: Damped oscillator."
        )
        self.figure.canvas.draw_idle()   

    def _open_nonlinear_pendulum(self, _event) -> None:
        """Open an independent nonlinear-pendulum interface."""

        model = build_nonlinear_pendulum_model()
        runner = InteractiveModelRunner(model)

        self.open_runners.append(runner)
        runner.figure.show()

        self.status_text.set_text(
            "Opened: Nonlinear pendulum."
        )
        self.figure.canvas.draw_idle()    

    def show(self) -> None:
        """Display the model catalogue."""

        plt.show()