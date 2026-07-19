"""Launch the dynuncertainty model catalogue."""

from dynuncertainty.catalog_interface import ModelCatalogWindow


def main() -> None:
    """Open the interactive dynuncertainty catalogue."""

    ModelCatalogWindow().show()


if __name__ == "__main__":
    main()