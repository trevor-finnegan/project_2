import numpy as np

def main():
    data = np.load("calibration_results.npz", allow_pickle=True)

    Rs = data["Rs"]
    Ts = data["Ts"]
    image_paths = data["image_paths"]

    print(f"Loaded {len(Rs)} extrinsic parameter sets.\n")

    for i, (R, T, path) in enumerate(zip(Rs, Ts, image_paths), start=1):
        print(f"=== Image #{i}: {path} ===")
        print("Rotation matrix R:")
        print(R)
        print("\nTranslation vector T:")
        print(T)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
