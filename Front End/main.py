"""
StyleSense: Retail Fashion Sales Prediction Dashboard
MBA Data Science & AI Project

Helper launcher script to automatically open index.html in your default web browser.
"""

import os
import sys
import webbrowser

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "index.html")

    if not os.path.exists(html_path):
        print(f"Error: index.html not found in {current_dir}")
        sys.exit(1)

    print("=" * 65)
    print(" StyleSense: Retail Fashion Sales Prediction Dashboard")
    print("=" * 65)
    print(f"Opening dashboard in your web browser:\nfile://{html_path}\n")
    print("Note: An internet connection is required to load TensorFlow.js and Chart.js CDNs.")
    print("=" * 65)

    webbrowser.open(f"file://{html_path}")

if __name__ == "__main__":
    main()

