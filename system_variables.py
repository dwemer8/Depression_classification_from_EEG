class SystemVariables:
    '''
    Getters for system variables for systems where I used notebook. Up to now there are Colab, Kaggle, Windows (my local machine) and Linux (lab server)
    '''
    def check_system_name(self, SYSTEM_NAME=None):
        if SYSTEM_NAME is None: SYSTEM_NAME = self.system_name
        if SYSTEM_NAME not in ["Colab", "Kaggle", "Windows", "Linux"]: raise NotImplementedError(f"Unkown system name: {SYSTEM_NAME}")
        else: pass
            
    def __init__(self, SYSTEM_NAME):
        self.check_system_name(SYSTEM_NAME)

        if SYSTEM_NAME == "Colab":
            from google.colab import drive
            drive.mount('/content/drive/')

        self.system_name = SYSTEM_NAME
        self.project_folder = self.get_project_folder()

    def get_system_name(self):
        return self.system_name

    def set_system_name(self):
        self.check_system_name(SYSTEM_NAME)
        return self.system_name

    def get_project_folder(self):
        # Project folder
        self.check_system_name()
        if self.system_name == "Windows": PROJECT_FOLDER = ""
        elif self.system_name == "Colab": PROJECT_FOLDER = "drive/MyDrive/Colab Notebooks/Neuroimaging and ML Group/Autoencoders/"  
        elif self.system_name == "Kaggle": PROJECT_FOLDER = "/kaggle/"  
        elif self.system_name == "Linux": PROJECT_FOLDER = ""
        
        return PROJECT_FOLDER

    def get_src_folder(self):
        self.check_system_name()
        if self.system_name in ["Windows", "Colab", "Linux"]: SRC_FOLDER = self.project_folder
        elif self.system_name == "Kaggle": SRC_FOLDER = self.project_folder + "input/eeg-age-prediction-utils/"
    
        return SRC_FOLDER

    def get_output_folder(self):
        self.check_system_name()
        if self.system_name in ["Windows", "Colab", "Linux"]: OUTPUT_FOLDER = self.project_folder
        elif self.system_name == "Kaggle": OUTPUT_FOLDER = self.project_folder + "working/"

        return OUTPUT_FOLDER

    def get_TUAB_folders(self):
        self.check_system_name()
        
        if self.system_name in ["Windows", "Colab", "Linux"]: TUAB_DIRECTORY = self.project_folder + "Data/TUAB/"
        elif self.system_name == "Kaggle": TUAB_DIRECTORY = self.project_folder + "input/tuab-age-prediction-60-s-all-records/TUAB/"
        
        if self.system_name in ["Windows", "Linux", "Colab"]: TUAB_TRAIN = TUAB_DIRECTORY + "train/normal/01_tcp_ar/"
        elif self.system_name in ["Kaggle"]: TUAB_TRAIN = TUAB_DIRECTORY + "train/"
        
        if self.system_name in ["Windows", "Linux", "Colab"]: TUAB_EVAL = TUAB_DIRECTORY + "eval/normal/01_tcp_ar/"
        elif self.system_name in ["Kaggle"]: TUAB_EVAL = TUAB_DIRECTORY + "eval/"

        return TUAB_DIRECTORY, TUAB_TRAIN, TUAB_EVAL

    def get_depr_anon_folder(self):
        self.check_system_name()
        if self.system_name in ["Windows", "Linux", "Colab"]: DEPR_ANON_DIRECTORY = self.project_folder + "Data/depression_anonymized/" #in microvolts
        elif self.system_name == "Kaggle": DEPR_ANON_DIRECTORY = self.project_folder + "input/depression-anonymized/"

        return DEPR_ANON_DIRECTORY

    def get_inhouse_folder(self):
        self.check_system_name()
        if self.system_name in ["Windows", "Linux", "Colab"]: INHOUSE_DIRECTORY = self.project_folder + "Data/inhouse_dataset/EEG_baseline_with_markers_cleaned/preprocessed_data/EEG_baseline/"
        elif self.system_name == "Kaggle": INHOUSE_DIRECTORY = self.project_folder + "input/inhouse-dataset/"

        return INHOUSE_DIRECTORY