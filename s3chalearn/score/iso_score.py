import os
import sys
import data_io

DEBUG = False


def main():

    try:
        # get the ref file
        ref_path = sys.argv[1]
        # get the res file
        res_path = sys.argv[2]
        # build ref dict
        gt_dict = {}
        pd_dict = {}
        with open(ref_path, 'r') as ref_open:
            for line in ref_open.readlines():
                m_file, k_file, label = line.split()
                assert m_file.replace('M_', 'K_') == k_file
                gt_dict[m_file] = int(label)
                assert len(gt_dict) > 0
        with open(res_path, 'r') as res_open:
            for line in res_open.readlines():
                m_file, k_file, label = line.split()
                assert m_file.replace('M_', 'K_') == k_file
                pd_dict[m_file] = int(label)
                assert len(pd_dict) > 0
        correct = 0.
        for m_file, label in pd_dict.items():
            if label == gt_dict[m_file]:
                correct += 1
        acc = float(correct) / len(gt_dict)
    except Exception as err:
        print err
        return
    score_result = open(sys.argv[3], 'wb')
    score_result.write("Accuracy: %0.6f\n" % acc)
    score_result.close()
    print acc
if __name__ == '__main__':
    main()

