"""
Utility script to test Kaggle API authentication.

Usage:
    python Utilities/System/kaggle_auth.py
"""
from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    print("Autenticado correctamente")


if __name__ == "__main__":
    main()
