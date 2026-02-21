import os
import shutil
import glob
import zipfile

def package_submission():
    base_dir = "submission_package"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    # 1. Source Code
    src_dir = os.path.join(base_dir, "src")
    os.makedirs(src_dir)
    
    # Copy simulation module
    shutil.copytree("simulation", os.path.join(src_dir, "simulation"), 
                    ignore=shutil.ignore_patterns("*.pyc", "__pycache__", "checkpoints", "logs", "outputs"))
    
    # Copy scripts
    shutil.copy("reproduce.py", src_dir)
    shutil.copy("requirements.txt", src_dir)
    shutil.copy("README.md", src_dir)

    # 2. Paper
    paper_dir = os.path.join(base_dir, "paper")
    os.makedirs(paper_dir)
    for f in glob.glob("paper/*.tex") + glob.glob("paper/*.bib") + glob.glob("paper/*.sty"):
        shutil.copy(f, paper_dir)
    # Copy figures (only png/pdf in root of paper dir if any? usually they are in simulation/outputs or similar)
    # But usually figures are in a subdirectory. Let's assume figures are needed.
    # Check if paper refers to images. Usually they are in simulation/outputs/reproduce/
    # We should copy them to paper/figures or similar if the tex expects them there.
    # Based on previous context, figures are in simulation/outputs/reproduce.
    
    # 3. Data
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir)
    # Copy JSON results
    for f in glob.glob("simulation/outputs/reproduce/*.json"):
        shutil.copy(f, data_dir)
    # Copy Figures for easy viewing
    figs_dir = os.path.join(base_dir, "figures")
    os.makedirs(figs_dir)
    for f in glob.glob("simulation/outputs/reproduce/*.png"):
        shutil.copy(f, figs_dir)

    # 4. Create README_SUBMISSION.md
    with open(os.path.join(base_dir, "README_SUBMISSION.md"), "w", encoding="utf-8") as f:
        f.write("# EthicaAI Submission Package\n\n")
        f.write("This package contains the source code, paper source, and experimental data for EthicaAI.\n\n")
        f.write("## Directory Structure\n")
        f.write("- `src/`: Python source code (JAX based simulation)\n")
        f.write("- `paper/`: LaTeX source files\n")
        f.write("- `data/`: Experimental result JSONs\n")
        f.write("- `figures/`: Generated plots\n\n")
        f.write("## Reproduction\n")
        f.write("To reproduce the results:\n")
        f.write("1. Install dependencies: `pip install -r src/requirements.txt`\n")
        f.write("2. Run reproduction script: `python src/reproduce.py --phase all`\n")

    # 5. Zip
    zip_filename = "ethicaai_submission_v2.zip"
    shutil.make_archive("ethicaai_submission_v2", 'zip', base_dir)
    
    print(f"Package created: {zip_filename}")
    print(f"Contents: {base_dir}/")

if __name__ == "__main__":
    package_submission()
