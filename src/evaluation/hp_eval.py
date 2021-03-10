
from src.result_storing import load_hp_results
from src.run_hp_tuning import adjust_hp_param

def eval_hp_results(result_path):
    dict_to_store = load_hp_results(result_path)
    print(dict_to_store)
    