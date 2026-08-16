import subprocess
import sys
import time

def run_step(step_name, command):
    print("\n" + "#" * 60)
    print(f" RUNNING: {step_name}")
    print("#" * 60)
    start_time = time.time()
    result = subprocess.run([sys.executable] + command)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"ERROR: {step_name} failed!")
        sys.exit(result.returncode)
    print(f"Completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    print("Starting Physics-Informed Preprocessing Pipeline...")
    
    run_step("Data Inspection", ["src/preprocess/01_inspect_data.py"])
    run_step("Alignment & Feature Fusion", ["src/preprocess/02_process_and_align.py"])
    run_step("Patch Extraction & Splitting", ["src/preprocess/03_extract_patches.py"])
    run_step("Compute Normalization Stats", ["src/preprocess/05_compute_normalization.py"])
    run_step("Dataset QA Verification", ["src/preprocess/06_verify_dataset.py"])

    print("You can now proceed to train the model Prithvi and Srijan")
    print(" Processed files saved in: data/processed/")