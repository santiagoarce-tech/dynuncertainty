"""Launch the interactive harmonic-oscillator example."""

from dynuncertainty.interface import InteractiveModelRunner
from dynuncertainty.models import build_harmonic_oscillator_model


def main() -> None:
    """Build the example model and launch its interface."""

    model = build_harmonic_oscillator_model()
    application = InteractiveModelRunner(model)
    application.show()


if __name__ == "__main__":
    main()