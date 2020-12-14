# import sys
# import uuid
# import time

def main():
    import octolib.helper as hp

    threshold_set = [1,2]
    detection_range_set = [0.2, 0.3, 0.5, 0.6]
    e_set = [15]
    ms_set = [15]
    cm_set = ["cityblock"]

    hp.automate_score_generation(
        threshold_set = threshold_set,
        detection_range_set = detection_range_set,
        e_set = e_set,
        ms_set = ms_set,
        cm_set = cm_set,
        mode = "grid_search"
        )


if __name__ == '__main__':
    # try:
        # sys.stdout = open("log/{}.log".format(uuid.uuid4()), "w")
        # print("[{}] # ".format(time.ctime()) + "Program Started. Logging Started")
    main()
        # sys.stdout.close()
    # except KeyboardInterrupt:
        # print("[{}] # ".format(time.ctime()) + "Interrupted by User. Closing Log and exiting.")
        # sys.stdout.close()
        # sys.exit(0)
