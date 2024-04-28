import pandas as pd

def get_runs(project, api=None):
    runs = api.runs(project)
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        name_list.append(run.name)
    
    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )
    return runs_df