import glob
import json
import os
import numpy as np
import argparse

def load_json(file):
    output = []
    for line in open(file, 'r'):
        output.append(json.loads(line))
    return output


np.set_printoptions(precision=2)

def decode_kitti_output(file, 
                        select_best=False, 
                        select_epoch=None,
                        min_epoch=15):
    results = load_json(file)

    result_list = {}
    for result in results[::-1]:
        if "mode" in result and \
            result["mode"] == "val":
            epoch = int(result["epoch"])
            if epoch <= min_epoch:
                continue
            result_list[epoch] = result
    if len(result_list) == 0:
        return None
    if select_best is False and select_epoch is None:
        # select last
        result = result_list[result_list.keys()[0]]
    elif select_epoch:
        result = result_list[int(select_epoch)]
    elif select_best:
        val_car_easy_ap = [float(item["1_KITTI/Car_3D_AP40_strict_easy"]) \
                    for key, item in result_list.items()]
        val_car_easy_ap = np.array(val_car_easy_ap)
        index = val_car_easy_ap.argmax()
        keys = list(result_list.keys())
        result = result_list[keys[index]]
    if "mode" not in result or result["mode"] != "val":
        return None
    output = dict()
    output["epoch"] = result["epoch"]
    output["iter"] = int(result["epoch"]) * int(result["iter"])
    for cat in ["Car", "Pedestrian", "Cyclist"]:
        # it means evaluate with train data:
        for diff in ["easy", "moderate", "hard"]:
            if f"0_KITTI/{cat}_3D_AP40_strict_easy" in result:
                output[f"train/{cat}_ap40_{diff}"] = \
                    float(result[f"0_KITTI/{cat}_3D_AP40_strict_{diff}"])
    for cat in ["Car", "Pedestrian", "Cyclist"]:
        # it means evaluate with train data:
        for diff in ["easy", "moderate", "hard"]:
            if f"1_KITTI/{cat}_3D_AP40_strict_easy" in result:
                output[f"val/{cat}_ap40_{diff}"] = \
                    float(result[f"1_KITTI/{cat}_3D_AP40_strict_{diff}"])   
    for cat in ["Car", "Pedestrian", "Cyclist"]:
        # it means evaluate with train data:
        for diff in ["easy", "moderate", "hard"]:
            if f"KITTI/{cat}_3D_AP40_strict_easy" in result:
                output[f"val/{cat}_ap40_{diff}"] = \
                    float(result[f"KITTI/{cat}_3D_AP40_strict_{diff}"])
    return output



def summarize_results(meta_name, dataset="kitti", start_time = 0, end_time = 12301230,
                     print_all = False,
                     select_best=False,
                     select_epoch=None,
                     min_epoch=15):
    print(f"---- {meta_name} -----")
    exps = glob.glob(f"{meta_name}/*")
    exps = [x[:-1] for x in exps]
    valid_exps = []
    for exp in exps:
        time_split = exp.split("/")[-1]
        time_split = "".join(time_split.split("_"))
        try:
            time_split = int(time_split)
        except:
            continue
        if time_split > start_time and time_split < end_time:
            valid_exps.append(exp)
    outputs = {}
    for exp in exps:
        if dataset == "kitti":
            exp_log = glob.glob(f"{exp}-/*.json")
            if len(exp_log) == 0:
                continue
            output = decode_kitti_output(exp_log[0],
                                        select_best=select_best,
                                        select_epoch=select_epoch,
                                        min_epoch=min_epoch)
            if output is not None:
                for key, item in output.items():
                    if key == "mode":
                        continue
                    if key not in outputs:
                        outputs[key] = []
                        outputs[key].append(item)
                    else:
                        outputs[key].append(item)
    
    for key, item in outputs.items():
        outputs[key] = np.array(item)
    # print(exps)
    if "epoch" in outputs:
        get_avg_results(outputs, meta_name, print_all=print_all,
                        select_best=select_best)
         
def get_avg_results(outputs, meta_name, print_all=False, select_best=False):
    epochs = set(outputs["epoch"])
    if not print_all:
        epochs = [max(epochs)]
    if select_best:
        epochs = [20]
    for epoch in epochs:
        selected_output = {}
        selected_output_all = {}
        if not select_best:
            valid_mask = outputs["epoch"]==epoch
        else:
            valid_mask = np.ones(len(outputs["epoch"])).astype(np.bool)
        for key, item in outputs.items():
            if key in ["mode", "epoch", "iter"]:
                continue
            selected_output[key] = item[valid_mask].mean()
            selected_output_all[key] = item[valid_mask]
        with open(f"{meta_name}/stat.txt", "a") as f:
            print(f"=========== epoch:\t{epoch}, \t exp times:\t{valid_mask.sum()} =============")
        for key, item in selected_output.items():
            with open(f"{meta_name}/stat.txt", "a") as f:
                f.write(f"epoch: {epoch}\n")
                print(f"{key}:  {item:.2f} \t {selected_output_all[key]}")
                f.write(f"{key}:  {item}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="summarize results")
    parser.add_argument("--meta_dir", default=None)
    parser.add_argument("--dataset", default="kitti")
    parser.add_argument("--start_time", default=0)
    parser.add_argument("--end_time", default=1230)
    parser.add_argument("--select_best", action="store_true")
    parser.add_argument("--select_epoch", default=None, type=int)
    parser.add_argument("--min_epoch", default=15, type=int)

    parser.add_argument("--print_all", action="store_true")

    args = parser.parse_args()

    if args.meta_dir == None:
        for meta_dir in glob.glob("*"):
            summarize_results(meta_dir, args.dataset,
                            args.start_time, args.end_time, args.print_all,
                            select_best=args.select_best,
                            select_epoch=args.select_epoch,
                            min_epoch=args.min_epoch)
    else:
        summarize_results(args.meta_dir, args.dataset,
                        args.start_time, args.end_time, args.print_all,
                            select_best=args.select_best,
                            select_epoch=args.select_epoch,
                            min_epoch=args.min_epoch)
