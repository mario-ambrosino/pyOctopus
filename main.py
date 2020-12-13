# import sys
# import uuid
# import time

def main():
    import octolib.helper as hp
    hp.test_alignment_score()


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
