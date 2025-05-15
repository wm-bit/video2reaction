import os
import json
from texttable import Texttable
import latextable


def get_mean(exp_name, seed_list=[193, 320, 502, 728, 846]):
    root_path = "/work/v2r/results"

    exp_names = [exp_name + f"_seed{s}" for s in seed_list]

    avg_test = {}

    for exp_name in exp_names:
        exp_path = os.path.join(root_path, exp_name)
        test_performance = json.load(open(os.path.join(exp_path, "test_performance.json")))["test_acc"]
        for key in test_performance:
            if key == "loss":
                continue

            if key not in avg_test:
                avg_test[key] = {}
            
            for metric in test_performance[key]:
                if metric not in avg_test[key]:
                    avg_test[key][metric] = [test_performance[key][metric]]
                else:
                    avg_test[key][metric].append(test_performance[key][metric])
    
    avg_test_mean = {}

    for key in avg_test:
        avg_test_mean[key] = {}
        for metric in avg_test[key]:
            avg_test_mean[key][metric] = sum(avg_test[key][metric]) / len(avg_test[key][metric])
    
    return avg_test, avg_test_mean


def __get_f1(results, fileds, loss):
    table = Texttable()
    heads = ["Model Name"]
    for field in fileds:
        heads.append(field.replace("_", "\_"))

    latex_rows = [heads]
    for exp_name in results:
        row = [exp_name.replace("_", "\_")]
        for field in fileds:
            if field in results[exp_name]["ldl"]:
                row.append(f"${results[exp_name]['ldl'][field]:.5f}$")
            else:
                row.append(f"${results[exp_name]['cls'][field]:.5f}$")
        latex_rows.append(row)
            
    table.add_rows(latex_rows)
    print('Texttable Output:')
    print(table.draw())
    print('\nLatextable Output:')
    print(latextable.draw_latex(table, caption=loss, label="table:f1"))


def get_f1_on_loss(loss):
    results = {
        "cten":  get_mean(f"cten_vaanet_average_{loss}")[1],
        "mminfomax": get_mean(f"mminfomax_average_{loss}")[1],
        "propose": get_mean(f"propose_tri_{loss}")[1]
    }
     
    __get_f1(results, ["chebyshev", "kl", "cad_ordinal_va", "intersection", "f1_weighted_0.1", "f1_weighted_0.3", "eec"], loss)


if __name__ == "__main__":
    # print("cten_vaanet_average_CrossEntropyLoss", get_mean("cten_vaanet_average_CrossEntropyLoss")[1])
    # print("mminfomax_average_CrossEntropyLoss", get_mean("mminfomax_average_CrossEntropyLoss")[1])
    # print("propose_average_CrossEntropyLoss", get_mean("propose_average_CrossEntropyLoss")[1])
    # print("propose_tri_CrossEntropyLoss",  get_mean("propose_tri_CrossEntropyLoss")[1])
   
   get_f1_on_loss("CECJSLoss")