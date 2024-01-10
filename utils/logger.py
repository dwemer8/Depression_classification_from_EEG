import os
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(
        self,
        log_type : str,
        model : torch.nn.Module = None,

        #saving
        save_path : str = None,
        model_name : str = None,

        #wandb/tensorboard
        run_name : str = None,
        
        #tensorboard
        log_dir : str = None,
        
        #wandb
        project_name : str = None,
        config : dict = {},
        log_freq : int = 10,
        model_description : str = None,
    ):
        
        self.log_type = log_type
        self.run_name = run_name
        self.model = model
        self.model_name = model_name
        self.save_path = save_path
        self.is_save_model = self.save_path is not None and self.model_name is not None
        self.reset()
        
        if self.log_type == "tensorboard":
            if self.log_dir is not None:
                print(f"Logging via TensorBoard, logs are in {self.log_dir}")
                
                self.log_dir = log_dir
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                os.system("%tensorboard --logdir=self.log_dir --port 6007")
                self.writer = SummaryWriter(self.log_dir)
                
            else:
                raise ValueError("Logging directory wasn't set")
                
        elif self.log_type == "wandb":
            print("Logging via WandB")
            
            self.run = wandb.init(name=self.run_name, project=project_name, config=config)  # Initialize wandb
            self.artifact = wandb.Artifact(config['model'], type='model', description=model_description, metadata=config)
            
            wandb.watch(model, log_freq=log_freq)
            
        elif self.log_type == "none":
            print("No logging")
        
        else:
            raise ValueError("Unknown logging type")
            
    def addOrCreate(self, d, k, v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        
        if k in d: d[k] += v
        else: d[k] = v
                
    def reset(self):
        self.per_step_values = {}
        self.n_steps = 0
        self.other_values = {}
        self.n_steps_other = 0
        
    def get(self):
        return {**self.per_step_values, **self.other_values}
        
    def _append(self, k, v):
        self.addOrCreate(self.per_step_values, k, v)
    
    def _append_other(self, k, v):
        self.addOrCreate(self.other_values, k, v)

    def flatten(self, d, current_path=""):
        def join_paths(root, leaf):
            if root == "": return leaf
            else: return root + "." + leaf

        d_flatten = {}
        
        for key in d:
            if isinstance(d[key], dict): d_flatten.update(self.flatten(d[key], current_path=join_paths(current_path, key)))
            elif isinstance(d[key], tuple) or isinstance(d[key], list):
                for i in range(len(d[key])):
                    d_flatten.update(self.flatten({f"{i}": d[key][i]}, join_paths(current_path, key)))
            else:
                d_flatten[join_paths(current_path, key)] = d[key]
                
        return d_flatten
                    
    def update(self, d):
        flatten_d = self.flatten(d)
        for key in flatten_d:
            self._append(key, flatten_d[key])
        self.n_steps += 1
    
    def update_other(self, d):
        flatten_d = self.flatten(d)
        for key in flatten_d:
            self._append_other(key, flatten_d[key])
        self.n_steps_other += 1
        
    def average(self):
        for key in self.per_step_values: self.per_step_values[key] /= self.n_steps
        for key in self.other_values: self.other_values[key] /= self.n_steps_other   
        
    def log(self, epoch_type, epoch):
        if self.log_type == "tensorboard":
            self.writer.add_scalar(self.run_name, scalar_value={epoch_type: self.get()}, global_step=epoch)
        
        elif self.log_type == "wandb":
            wandb.log({epoch_type: self.get()}, step=epoch)
            
    def _log_model(self, model_file):
        self.artifact.add_file(model_file)
            
    def save_model(self, epoch):
        if self.log_type == "none":
            pass
        
        elif self.is_save_model:
            save_dir = os.path.join(self.save_path, self.model_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            model_file = os.path.join(save_dir, str(epoch) + "_epoch.pth")
            torch.save(self.model.state_dict(), model_file)

            if self.log_type == "wandb":
                self._log_model(model_file)
        
        elif self.save_path is not None and self.model_name is None:
            raise ValueError("Model_name hasn't been set whereas save_path has")
        
    def finish(self):
        if self.log_type == "tensorboard": 
            self.writer.close()
        
        if self.log_type == "wandb":
            if self.is_save_model:
                self.run.log_artifact(self.artifact)
            wandb.finish()

    def update_summary(self, key, value):
        if self.log_type == "wandb":
            self.run.summary[key] = value
            # self.run.summary.update()
        else:
            print(f"WARNING: log type should be 'wandb' instead of {self.log_type}, nothing was done")