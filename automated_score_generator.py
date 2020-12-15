"""
Script to automate the score comparison with a simple grid search.

TODO EXTEND TUTORIAL:

Example of usage:

python3 automated_score_generator.py --THR "0.1; 0.5" --DER "0.1; 0.4" --EPS "15" --MIS "15" --CME "cityblock ; manhattan"


"""

import time
import argparse
import lib.helper as hp

if __name__ == '__main__':
    try:
        # Start Logging
        # sys.stdout = open("log/{}.log".format(uuid.uuid4()), "w")
        print("[{}] # ".format(time.ctime()) + "Logging Started")

        # Parse Input
        parser = argparse.ArgumentParser()
        parser.add_argument("--THR", type = str, default = 0, help = "list(float): Threshold - from 0.1 to 3")
        parser.add_argument("--DER", type = str, default = 0, help = "list(float): Detection Range [m] - from 0.1 to 5")
        parser.add_argument("--EPS", type = str, default = 0, help = "list(int): Epsilon-Neighborhood for DBSCAN")
        parser.add_argument("--MIS", type = str, default = 0, help = "list(int): Min Samples for DBSCAN")
        parser.add_argument("--CME", type = str, default = 0, help = "list(strings): Metric to choose for DBSCAN")
        opt = parser.parse_args()

        # Assign Grid delimiters
        threshold_set = [float(s) for s in opt.THR.replace(" ", "").split(";")]
        detection_range_set = [float(s) for s in opt.DER.replace(" ", "").split(";")]
        e_set = [float(s) for s in opt.EPS.replace(" ", "").split(";")]
        ms_set = [float(s) for s in opt.MIS.replace(" ", "").split(";")]
        cm_set = [str(s) for s in opt.CME.replace(" ", "").split(";")]

        print("[{}] # ".format(time.ctime()) + "Script Automated Score Generator Started. \n\nPARAMETERS :: \nTHR:"
                                               " {}\nDER: {}\nEPS: {}\nMIS: {}\nCME: {}\n".format(
            threshold_set, detection_range_set, e_set, ms_set, cm_set
            ))

        # Generate Scores
        hp.automate_score_generation(
            threshold_set = threshold_set,
            detection_range_set = detection_range_set,
            e_set = e_set,
            ms_set = ms_set,
            cm_set = cm_set,
            mode = "grid_search"
            )

        # sys.stdout.close()
    except KeyboardInterrupt:
        print("[{}] # ".format(time.ctime()) + "Interrupted by User. Closing Log and exiting.")
        # sys.stdout.close()
        # sys.exit(0)
