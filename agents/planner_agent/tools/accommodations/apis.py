import os
import sys

import pandas as pd
from pandas import DataFrame
from typing import Optional

from agents.planner_agent.utils.func import extract_before_parenthesis
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# from utils.func import extract_before_parenthesis


class Accommodations:
    def __init__(self, path=None):
        if path is None:
            # 1. Get the folder containing apis.py
            this_dir = os.path.dirname(os.path.abspath(__file__))
            # 2. Build the path to the CSV relative to apis.py
            path = os.path.join(
                this_dir,
                "..", "..",  # up two levels: from tools/flights/ to planner_agent/
                "database",
                "accommodations",
                "clean_accommodations_2022.csv"
            )
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['NAME','price','room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']]
        print("Accommodations loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == city]
        if len(results) == 0:
            return "There is no attraction in this city."
        
        return results
    
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == extract_before_parenthesis(city)]
        return results