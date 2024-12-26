import os
import sys

import pandas as pd
from pandas import DataFrame
from typing import Optional
from agents.planner_agent.utils.func import extract_before_parenthesis
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Attractions:
    def __init__(self, path=None):
        if path is None:
            # 1. Get the folder containing apis.py
            this_dir = os.path.dirname(os.path.abspath(__file__))
            # 2. Build the path to the CSV relative to apis.py
            path = os.path.join(
                this_dir,
                "..", "..",  # up two levels: from tools/flights/ to planner_agent/
                "database",
                "attractions",
                "attractions.csv"
            )
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['Name','Latitude','Longitude','Address','Phone','Website',"City"]]
        print("Attractions loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There is no attraction in this city."
        return results  
      
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        # the results should show the index
        results = results.reset_index(drop=True)
        return results