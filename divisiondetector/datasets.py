import os
import pandas as pd
from torch.utils.data import Dataset
from utils.gp_pipeline import GPPipeline

class DivisionDataset(Dataset):
    def __init__(self, label_path, img_path, window_size=(100,100,100), time_window=(1, 1), mode='ball', ball_radius=(10, 10, 10)):
        def __getlabels(label_path, div_path):
            '''
            Load and process label data
            '''
            raw_df = pd.read_csv(label_path, encoding='unicode_escape')

            print("Data loaded.")

            # Converting and formatting
            columns = ["Timepoint", "X", "Y", "Z"]

            raw_df = raw_df.loc[2:].drop(["Label", "Detection quality"], axis=1) # Dropping unnecessary columns
            raw_df["ID"] = raw_df["ID"].astype(int) # Ensure the IDs are integers
            raw_df = raw_df.set_index("ID") # Set index to ID

            df = raw_df.apply(lambda x: pd.Series([int(x[0])] + [float(element) for element in x[1:]], index=columns), axis=1) # Convert coordinates and timepoints to numbers (from strings) and relabel
            # df["Z"] = df["Z"] / 5 # Made redundant by voxel_size

            print("Data processed.")

            # Gunpowder wants its CSVs separated with ', ' and not just ','
            df = df.apply(lambda x: pd.Series([str(element) for element in x], index=columns), axis=1)
            columns = ["Timepoint", "Z", "Y", "X"] # Reorder columns
            df["Timepoint"] = df["Timepoint"].astype(int)
            df[columns].assign(id=df.index.to_series()).to_csv(div_path, sep=' ', index=False, header=False)

            print("Data written.")

            return df

        def __getCSV(label_path, div_path):
            if os.path.isfile(div_path):
                df = pd.read_csv(div_path, sep='\, ', header=None, engine='python')
                df.columns = ["Timepoint", "Z", "Y", "X", "ID"]
                df = df.set_index("ID")

                print("File exists. Data loaded.")

                return df
            else:
                return __getlabels(label_path, div_path)

        print("Initialising...")

        div_path = "division.csv"

        self.labels = __getCSV(label_path, div_path)
        self.img_path = img_path
        self.window_size = window_size
        self.time_window = time_window


        self.pipeline = GPPipeline(img_path, div_path, mode, ball_radius)

        print("Pipeline created.")

    def __len__(self):
        print(self.labels["Z"].max())
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        5D-array
        '''
        data = self.labels.loc[idx]

        c_t_vol, target, points = self.pipeline.fetch_data(
            (data["Z"], data["Y"], data["X"]),
            self.window_size,
            data["Timepoint"],
            self.time_window
        )

        return c_t_vol, target, points