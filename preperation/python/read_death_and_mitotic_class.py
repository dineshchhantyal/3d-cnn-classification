from pathlib import Path
import pandas as pd

def read_death_and_mitotic_class(path):
    """Load and summarize classification data from DeathAndMitoticClass.txt"""
    p = Path(path)
    class_file = p / "DeathAndMitoticClass.txt"

    if not class_file.exists():
        print(f"❌ File not found: {class_file}")
        return {}

    try:
        print(f"📂 Loading classification data from: {class_file}")

        # Load data with pandas - handles irregular spacing
        df = pd.read_csv(
            class_file,
            sep="\s+",
            header=None,
            names=["frame", "nucleus_id", "mitotic", "death"],
        )

        print(f"✅ Loaded {len(df)} classifications")
        print(f"Columns: {list(df.columns)}")

        # Show data distribution
        print(f"\nData distribution:")
        print(f"Frames: {df['frame'].min()} - {df['frame'].max()}")
        print(f"Nucleus IDs: {df['nucleus_id'].nunique()} unique")
        print(f"Mitotic events: {df['mitotic'].sum()}")
        print(f"Death events: {df['death'].sum()}")

        return {"classes": df}

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return {}
