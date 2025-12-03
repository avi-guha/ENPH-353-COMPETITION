#!/usr/bin/env python3
import os
import subprocess

def main():
    sim_script = "/home/fizzer/ENPH-353-COMPETITION/src/enph353/enph353_utils/scripts/run_sim.sh"

    # Ensure the script is executable
    os.chmod(sim_script, 0o755)

    # Launch simulation detached (non-blocking)
    subprocess.Popen(["bash", sim_script])

if __name__ == "__main__":
    main()
